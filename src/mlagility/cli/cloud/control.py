"""
WARNING: this is a work-in-progress tool. We do not recommend using it yet.
"""

import os
import argparse
import subprocess
import shutil
import time
from typing import Optional, Callable
import concurrent.futures
from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient

if os.environ.get("MLAGILITY_AZURE_LOCATION"):
    LOCATION = os.environ.get("MLAGILITY_AZURE_LOCATION")
else:
    LOCATION = "eastus"

USERNAME = "azureuser"

device_to_image_reference = {
    "nvidia": {
        "publisher": "nvidia",
        "offer": "ngc_azure_17_11",
        "sku": "ngc-base-version-23_03_0_gen2",
        "version": "latest",
    },
    "x86": {
        "publisher": "canonical",
        "offer": "0001-com-ubuntu-server-focal",
        "sku": "20_04-lts-gen2",
        "version": "latest",
    },
}

hardware_to_device = {
    "t4": "nvidia",
    "icelake": "x86",
    "cpu-small": "x86",
    "cpu-big-ram": "x86",
}

hardware_to_vm_size = {
    "t4": "Standard_NC4as_T4_v3",  # Nvidia T4 with 4 EPYC vCPUs
    "icelake": "Standard_D16ds_v5",  # 16-core Xeon IceLake with 64 GB RAM
    "cpu-small": "Standard_E2s_v3",  # 2-core Xeon (unknown generation) with 16 GB RAM
    "cpu-big-ram": "Standard_E16-4s_v3",  # 4-core Xeon with 128 GB RAM
}

vm_size_to_hardware = {v: k for k, v in hardware_to_vm_size.items()}

device_to_plan = {
    "nvidia": {
        "name": "ngc-base-version-23_03_0_gen2",
        "product": "ngc_azure_17_11",
        "publisher": "nvidia",
    },
    "x86": None,
}


def benchit_prefix(args: str) -> str:
    return f"miniconda3/bin/conda run -n mla benchit {args}"


def local_command(command: str):
    subprocess.run(command, check=True, shell=True)


def auth():
    # Acquire a credential object using CLI-based authentication.
    credential = AzureCliCredential()

    # Retrieve subscription ID from environment variable.
    subscription_id = os.environ["AZURE_SUBSCRIPTION_ID"]

    return credential, subscription_id


def create_resource_group(resource_client: ResourceManagementClient, rg_name: str):
    # Step 1: Provision a resource group

    # Provision the resource group.
    rg_result = resource_client.resource_groups.create_or_update(
        rg_name, {"location": LOCATION}
    )

    print(
        f"Provisioned resource group {rg_result.name} in the \
    {rg_result.location} region"
    )

    return rg_result


def create_network_security_group(
    network_client: NetworkManagementClient, nsg_name: str, rg_name: str
):

    poller = network_client.network_security_groups.begin_create_or_update(
        rg_name,
        nsg_name,
        {
            "location": LOCATION,
        },
    )

    _ = poller.result()

    # Create Security Rules
    # NOTE: These are a copy of the default security rules every VM gets
    # in Azure portal. May want to customize eventually.
    poller = network_client.security_rules.begin_create_or_update(
        rg_name,
        nsg_name,
        f"{nsg_name}-JupyterHub",
        security_rule_parameters={
            "protocol": "TCP",
            "sourcePortRange": "*",
            "destinationPortRange": "8000",
            "sourceAddressPrefix": "*",
            "destinationAddressPrefix": "*",
            "access": "Allow",
            "priority": 1020,
            "direction": "Inbound",
            "sourcePortRanges": [],
            "destinationPortRanges": [],
            "sourceAddressPrefixes": [],
            "destinationAddressPrefixes": [],
        },
    )

    jupyter_rule = poller.result()

    poller = network_client.security_rules.begin_create_or_update(
        rg_name,
        nsg_name,
        f"{nsg_name}-RStudio_Server",
        security_rule_parameters={
            "protocol": "TCP",
            "sourcePortRange": "*",
            "destinationPortRange": "8787",
            "sourceAddressPrefix": "*",
            "destinationAddressPrefix": "*",
            "access": "Allow",
            "priority": 1030,
            "direction": "Inbound",
            "sourcePortRanges": [],
            "destinationPortRanges": [],
            "sourceAddressPrefixes": [],
            "destinationAddressPrefixes": [],
        },
    )

    rstudio_rule = poller.result()

    poller = network_client.security_rules.begin_create_or_update(
        rg_name,
        nsg_name,
        f"{nsg_name}-SSH",
        security_rule_parameters={
            "protocol": "TCP",
            "sourcePortRange": "*",
            "destinationPortRange": "22",
            "sourceAddressPrefix": "*",
            "destinationAddressPrefix": "*",
            "access": "Allow",
            "priority": 1010,
            "direction": "Inbound",
            "sourcePortRanges": [],
            "destinationPortRanges": [],
            "sourceAddressPrefixes": [],
            "destinationAddressPrefixes": [],
        },
    )

    ssh_rule = poller.result()

    # Update security group with rules

    poller = network_client.network_security_groups.begin_create_or_update(
        rg_name,
        nsg_name,
        parameters={
            "location": LOCATION,
            "securityRules": [
                jupyter_rule,
                rstudio_rule,
                ssh_rule,
            ],
        },
    )

    nsg = poller.result()

    return nsg


def create_vnet_subnet_nsg(
    network_client: NetworkManagementClient,
    vnet_name: str,
    subnet_name: str,
    nsg_name: str,
    rg_name: str,
):

    nsg_result = create_network_security_group(network_client, nsg_name, rg_name)

    # A virtual machine requires a network interface client (NIC). A NIC
    # requires a virtual network and subnet along with an IP address.
    # Therefore we must provision these downstream components first, then
    # provision the NIC, after which we can provision the VM.

    poller = network_client.virtual_networks.begin_create_or_update(
        rg_name,
        vnet_name,
        {
            "location": LOCATION,
            "address_space": {"address_prefixes": ["10.0.0.0/16"]},
        },
    )

    vnet_result = poller.result()

    print(
        f"Provisioned virtual network {vnet_result.name} with address \
    prefixes {vnet_result.address_space.address_prefixes}"
    )

    # Step 3: Provision the subnet and wait for completion
    poller = network_client.subnets.begin_create_or_update(
        rg_name,
        vnet_name,
        subnet_name,
        {"address_prefix": "10.0.0.0/24"},
    )
    subnet_result = poller.result()

    print(
        f"Provisioned virtual subnet {subnet_result.name} with address \
    prefix {subnet_result.address_prefix}"
    )

    return vnet_result, subnet_result, nsg_result


class Device:
    """
    Base class for handling an individual device (VM). Can be inherited
    to provide more advanced functionality, for example a Cluster (device
    comprised of many sub-devices).
    """

    def __init__(
        self,
        name,
        retrieve: bool,
        hardware: Optional[str] = None,
        rg_name: Optional[str] = None,
        nsg_name: Optional[str] = None,
        vnet_name: Optional[str] = None,
        subnet_name: Optional[str] = None,
    ):
        # The names for the resource group, vnet, subnet, and nsg may be
        # set by a Cluster that are instantiating this Device
        if rg_name is None:
            self.RG_NAME = f"{name}-rg"
        else:
            self.RG_NAME = rg_name

        if vnet_name is None:
            self.VNET_NAME = f"{name}-vnet"
        else:
            self.VNET_NAME = vnet_name

        if subnet_name is None:
            self.SUBNET_NAME = f"{name}-subnet"
        else:
            self.SUBNET_NAME = subnet_name

        if nsg_name is None:
            self.NSG_NAME = f"{name}-nsg"
        else:
            self.NSG_NAME = nsg_name

        self.VM_NAME = f"{name}"
        self.IP_NAME = f"{name}-ip"
        self.IP_CONFIG_NAME = f"{name}-ip-config"
        self.NIC_NAME = f"{name}-nic"

        self.credential, self.subscription_id = auth()
        self.resource_client = ResourceManagementClient(
            self.credential, self.subscription_id
        )
        self.compute_client = ComputeManagementClient(
            self.credential, self.subscription_id
        )
        self.network_client = NetworkManagementClient(
            self.credential, self.subscription_id
        )

        self.async_process = None
        self.async_command = None

        self._ip = None

        self.hardware = hardware
        self.device = hardware_to_device[self.hardware]

        self.populate_devices(retrieve)

    def populate_devices(self, retrieve: bool):

        if retrieve:
            vm_size, _ = self.info(verbosity="none")
            vm_hardware = vm_size_to_hardware[vm_size]
            if self.hardware is not None and vm_hardware != self.hardware:
                raise ValueError(
                    f"Device initialized with hardware {self.hardware} but "
                    f"VM {self.VM_NAME} has hardware {vm_hardware}."
                )
            else:
                self.hardware = vm_hardware

    @property
    def ip_address(self):
        if not self._ip:
            self._ip = self.network_client.public_ip_addresses.get(
                self.RG_NAME, self.IP_NAME
            ).ip_address

        return self._ip

    def add_to_known_hosts(self):
        print(f"Adding {self.VM_NAME} ({self.ip_address}) to known hosts file")
        known_host = subprocess.check_output(
            ["ssh-keyscan", "-H", self.ip_address]
        ).decode(encoding="utf-8")
        known_hosts_file = os.path.expanduser("~/.ssh/known_hosts")
        with open(known_hosts_file, "a") as f:
            f.write(known_host)

    def info(self, verbosity="high"):
        vm_info = self.compute_client.virtual_machines.get(
            self.RG_NAME, self.VM_NAME, expand="instanceView"
        )

        status = vm_info.instance_view.statuses[1].display_status
        size = vm_info.hardware_profile.vm_size

        if verbosity == "high":
            print(f"{self.VM_NAME} ({size}, {self.ip_address}): {status}")

        return size, status

    def setup(self):
        # Add the VM IP to known hosts, and retry a few times before giving up
        self.retry_method(self.add_to_known_hosts, "Adding to known hosts")

        self.send_file("setup_part_1.sh")
        self.send_file("setup_part_2.sh")
        self.run_command("bash setup_part_1.sh")
        self.run_command("bash setup_part_2.sh")

    def create(
        self,
        create_shared_resources: bool = True,
        subnet_result=None,
        nsg_result=None,
    ):

        """
        Set create_shared_resources = False if this VM is part of a cluster,
        since those shared resources will be created at the cluster level.
        """

        if os.environ.get("MLAGILITY_AZURE_PASSWORD"):
            PASSWORD = os.environ.get("MLAGILITY_AZURE_PASSWORD")
        else:
            raise ValueError(
                "You must set the MLAGILITY_AZURE_PASSWORD environment variable "
                "to create VM(s). This will be the admin password for the VM(s)."
            )

        if os.environ.get("MLAGILITY_AZURE_PUBLIC_KEY"):
            ssh_key = os.environ.get("MLAGILITY_AZURE_PUBLIC_KEY")
        else:
            raise ValueError(
                "You must set the MLAGILITY_AZURE_PUBLIC_KEY environment variable "
                "to create VM(s). This will be the public ssh key added to the VM."
            )

        if create_shared_resources:
            create_resource_group(self.resource_client, self.RG_NAME)

            _, subnet_result, nsg_result = create_vnet_subnet_nsg(
                self.network_client,
                self.VNET_NAME,
                self.SUBNET_NAME,
                self.NSG_NAME,
                self.RG_NAME,
            )
        else:

            # Check and make sure the shared resources exist
            group_list = self.resource_client.resource_groups.list()
            resource_group_names = [item.name for item in group_list]

            if self.RG_NAME not in resource_group_names:
                raise ValueError(
                    f"Attempted to retrieve a resource group {self.RG_NAME} that "
                    "is not part of your Azure subscription. Found resource "
                    f"groups: {resource_group_names}"
                )

            if subnet_result is None or nsg_result is None:
                raise ValueError(
                    "Attempted to call VM.create with create_shared_resources=False, however "
                    f"subnet_result is {subnet_result} and nsg_result is {nsg_result}."
                    "Both are required to be populated (ie, not None)."
                )

        # Step 4: Provision an IP address and wait for completion
        poller = self.network_client.public_ip_addresses.begin_create_or_update(
            self.RG_NAME,
            self.IP_NAME,
            {
                "location": LOCATION,
                "sku": {"name": "Standard"},
                "public_ip_allocation_method": "Static",
                "public_ip_address_version": "IPV4",
            },
        )

        ip_address_result = poller.result()

        print(
            f"Provisioned public IP address {ip_address_result.name} \
        with address {ip_address_result.ip_address}"
        )

        # Step 5: Provision the network interface client
        poller = self.network_client.network_interfaces.begin_create_or_update(
            self.RG_NAME,
            self.NIC_NAME,
            {
                "location": LOCATION,
                "ip_configurations": [
                    {
                        "name": self.IP_CONFIG_NAME,
                        "subnet": {"id": subnet_result.id},
                        "public_ip_address": {"id": ip_address_result.id},
                    }
                ],
                "network_security_group": {"id": nsg_result.id},
            },
        )

        nic_result = poller.result()
        print("Provisioned NIC", nic_result.name)

        vm_spec = {
            "location": LOCATION,
            "storage_profile": {
                "image_reference": device_to_image_reference[self.device],
                "os_disk": {
                    "create_option": "FromImage",
                    "deleteOption": "Delete",
                },
            },
            "hardware_profile": {"vm_size": hardware_to_vm_size[self.hardware]},
            "os_profile": {
                "computer_name": self.VM_NAME,
                "admin_username": USERNAME,
                "admin_password": PASSWORD,
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": f"/home/{USERNAME}/.ssh/authorized_keys",
                                "key_data": ssh_key,
                            }
                        ]
                    },
                },
            },
            "network_profile": {
                "network_interfaces": [
                    {
                        "id": nic_result.id,
                    }
                ]
            },
        }

        if device_to_plan[self.device]:
            vm_spec["plan"] = device_to_plan[self.device]

        poller = self.compute_client.virtual_machines.begin_create_or_update(
            self.RG_NAME,
            self.VM_NAME,
            vm_spec,
        )

        vm_result = poller.result()
        print(f"Provisioned virtual machine {vm_result.name}")

    def delete(self):
        """
        Deletes the resources associated with a specific VM. Note: this may leave behind
        a resource group and other artifacts, so generally it is a good idea to go to
        portal.azure.com and finish cleaning up there.

        We don't delete the resource group when deleting a single VM since we don't want
        to make assumptions about what else is in that resource group.
        """

        print("Deleting VM", self.VM_NAME)
        # NOTE: we call result() to make sure the delete operation finishes before we move on to the next one
        self.compute_client.virtual_machines.begin_delete(
            resource_group_name=self.RG_NAME, vm_name=self.VM_NAME
        ).result()

        print("Deleting nic", self.NIC_NAME)
        self.network_client.network_interfaces.begin_delete(
            self.RG_NAME, self.NIC_NAME
        ).result()
        print("Deleting ip", self.IP_NAME)
        self.network_client.public_ip_addresses.begin_delete(
            self.RG_NAME, self.IP_NAME
        ).result()
        print("Deleting subnet", self.SUBNET_NAME)
        self.network_client.subnets.begin_delete(
            self.RG_NAME, self.VNET_NAME, self.SUBNET_NAME
        ).result()
        print("Deleting vnet", self.VNET_NAME)
        self.network_client.virtual_networks.begin_delete(
            self.RG_NAME, self.VNET_NAME
        ).result()

    def stop(self):
        print("Stopping VM", self.VM_NAME)
        poller = self.compute_client.virtual_machines.begin_deallocate(
            self.RG_NAME, self.VM_NAME
        )
        poller.result()

    def start(self):
        print("Starting VM", self.VM_NAME)
        poller = self.compute_client.virtual_machines.begin_start(
            self.RG_NAME, self.VM_NAME
        )
        poller.result()
        self.wait_ssh()

    def send_file(self, file):
        command = f"scp {file} {USERNAME}@{self.ip_address}:~/."
        print(f"Running command on {self.VM_NAME}: {command}")
        subprocess.run(command.split(" "), check=True)

    def get_file(self, remote_file, local_dir, local_file_name):
        os.makedirs(local_dir, exist_ok=True)
        command = f"scp {USERNAME}@{self.ip_address}:{remote_file} {local_dir}/{local_file_name}"
        print(f"Running command on {self.VM_NAME}: {command}")
        subprocess.run(command.split(" "), check=True)

    def get_dir(self, remote_dir, local_destination):
        # NOTE: removes anything currently at local_destination!
        if os.path.isdir(local_destination):
            shutil.rmtree(local_destination)

        os.makedirs(local_destination)

        # Tar the remote directory to make the transfer faster
        cache_tar_filename = "cache.tar.gz"
        self.run_command(f"tar -czvf {cache_tar_filename} {remote_dir}")

        # Copy the tar file
        command = (
            f"scp {USERNAME}@{self.ip_address}:{cache_tar_filename} {local_destination}"
        )
        print(f"Running command on {self.VM_NAME}: {command}")
        subprocess.run(command.split(" "), check=True)

        # Untar
        local_cache_tar_path = os.path.join(local_destination, cache_tar_filename)
        local_command(f"tar -xzvf {local_cache_tar_path} -C {local_destination}")

        # Clean up the tar file
        local_command(f"rm {local_cache_tar_path}")

    def run_command(self, command):
        full_command = f"ssh {USERNAME}@{self.ip_address} {command}"
        print(f"Running command on {self.VM_NAME}: {full_command}")
        subprocess.run(full_command.split(" "), check=True)

    def check_command(self, command, success_term):
        full_command = f"ssh {USERNAME}@{self.ip_address} {command}"
        print(f"Running command on {self.VM_NAME}: {full_command}")
        result = subprocess.check_output(full_command.split(" ")).decode()
        if success_term in result:
            print(f"\n\nSuccess on {self.VM_NAME}!\n\n")
        else:
            raise Exception("Error! Success term not in result")

    def run_async_command(self, command):
        full_command = f"ssh {USERNAME}@{self.ip_address} {command}"
        print(f"Running async command on {self.VM_NAME}: {full_command}")
        self.async_process = subprocess.Popen(
            full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.async_command = full_command

    def poll_async_command(self, verbosity: str = "low"):
        """
        Checks whether an async command has finished
        If the command has finished, resets self.async_process and returns the output
        Otherwise, returns None
        """

        if self.async_process.poll() is not None:
            out, errs = self.async_process.communicate()
            self.async_process = None
            if verbosity == "low" or verbosity == "high":
                print(f"{self.VM_NAME} finished job: {self.async_command}")

            if verbosity == "high":
                print(out.decode())
                print(errs.decode())
            return out, errs
        else:
            return None

    def finish_async_command(
        self, check_output: Optional[str] = None, verbosity: str = "low"
    ):
        """
        Waits for an async command to finish and then prints the result
        """
        out, errs = self.async_process.communicate()
        out = out.decode()
        errs = errs.decode()

        print(f"{self.VM_NAME} finished job: {self.async_command}")

        if verbosity == "high":
            print(out)
            print(errs)

        if check_output is not None:
            if check_output not in out:
                raise ValueError(f"Error! Success term not in result: {out}")
            else:
                print("Success!")

        self.async_process = None

        return out, errs

    def retry_method(
        self,
        method: Callable,
        task_description: str,
        timeout_seconds=300,
        sleep_amount_seconds=15,
    ):
        """
        Run a function, and if it doesn't work then retry until a timeout runs out
        """

        success_achieved = False

        if timeout_seconds % sleep_amount_seconds != 0:
            raise ValueError(
                f"sleep_amount_seconds ({sleep_amount_seconds}) must divide "
                f"evenly into timeout_seconds ({timeout_seconds})"
            )

        timeout_remaining = timeout_seconds

        while not success_achieved and timeout_remaining > 0:
            try:
                method()
                success_achieved = True
            except subprocess.CalledProcessError as e:
                timeout_remaining = timeout_remaining - sleep_amount_seconds
                if timeout_remaining > 0:
                    print(
                        f"{task_description} failed. "
                        f"Re-attempting, timeout remaining is {timeout_remaining}"
                    )
                else:
                    print(f"{task_description} failed and the timeout has elapsed.")
                    raise e
                time.sleep(sleep_amount_seconds)

    def ssh_hello_world(self):
        self.run_command("echo hello world")

    def wait_ssh(self):
        """
        Wait until the VM is available for SSH connection
        """

        self.retry_method(self.ssh_hello_world, "Checking SSH availability")

    def selftest(self):
        self.check_command(
            benchit_prefix(
                f"mlagility/models/selftest/linear.py --device {self.device}"
            ),
            "Successfully benchmarked",
        )
        self.wipe_mlagility_cache()

    def wipe_models_cache(self):
        # NOTE: this method needs to be kept up-to-date with MLAgility's corpus
        # such that it deletes all cached model files on disk. Otherwise the
        # VM disks will fill up.
        command = "rm -rf .cache/huggingface .cache/torch-hub"
        self.run_command(command)

    def wipe_mlagility_cache(self):
        command = benchit_prefix("cache delete --all")
        self.run_command(command)


class Cluster(Device):
    """
    Handle for controlling a cluster of VMs
    """

    def __init__(
        self,
        name: str,
        retrieve: bool,
        size: Optional[int] = None,
        hardware: str = "cpu-small",
        rg_name: Optional[str] = None,
        nsg_name: Optional[str] = None,
        vnet_name: Optional[str] = None,
        subnet_name: Optional[str] = None,
    ):

        self.size = size
        self.name = name

        self.vnet = None
        self.subnet = None
        self.nsg = None

        # self.devices will be populated with a list of sub-devices (e.g., VMs)
        self.devices = None

        super().__init__(
            name=name,
            retrieve=retrieve,
            hardware=hardware,
            rg_name=rg_name,
            nsg_name=nsg_name,
            vnet_name=vnet_name,
            subnet_name=subnet_name,
        )

    def populate_devices(self, retrieve: bool):
        if retrieve:
            resource_group_vms = self.compute_client.virtual_machines.list(self.RG_NAME)
            self.devices = [
                Device(
                    vm.name,
                    retrieve=True,
                    hardware=self.hardware,
                    rg_name=self.RG_NAME,
                    nsg_name=self.NSG_NAME,
                    vnet_name=self.VNET_NAME,
                    subnet_name=self.SUBNET_NAME,
                )
                for vm in resource_group_vms
                if vm.name.startswith(self.name)
            ]
        else:
            if self.size is None:
                raise ValueError("size must be set when retrieve=False")

            self.devices = [
                Device(
                    name=f"{self.name}-{i}",
                    retrieve=False,
                    hardware=self.hardware,
                    rg_name=self.RG_NAME,
                    nsg_name=self.NSG_NAME,
                    vnet_name=self.VNET_NAME,
                    subnet_name=self.SUBNET_NAME,
                )
                for i in range(self.size)
            ]

    def check_vms_exist(self):
        resource_group_vm_names = [
            vm.name for vm in self.compute_client.virtual_machines.list(self.RG_NAME)
        ]
        for vm in self.devices:
            if vm.VM_NAME not in resource_group_vm_names:
                raise ValueError(
                    f"VM {vm.VM_NAME} expected to be in group {self.RG_NAME}, "
                    "but was not found. We suggest checking to see if you are out "
                    f"of quota for {hardware_to_vm_size[vm.hardware]}."
                )

    def parallelize(self, method: Callable, **kwargs):
        """
        Run a method on all VMs in parallel
        """
        method_name = method.__name__
        futures = []
        with concurrent.futures.ThreadPoolExecutor(self.size) as executor:
            for device in self.devices:
                method_ref = getattr(device, method_name)
                if kwargs is None:
                    futures.append(executor.submit(method_ref))
                else:
                    futures.append(executor.submit(method_ref, **kwargs))

            concurrent.futures.wait(futures)

    def info(self, verbosity: str = "high"):
        self.parallelize(super().info, verbosity=verbosity)

    def create(
        self, create_shared_resources: bool = True, subnet_result=None, nsg_result=None
    ):
        if create_shared_resources:
            create_resource_group(self.resource_client, self.RG_NAME)
            self.vnet, self.subnet, self.nsg = create_vnet_subnet_nsg(
                self.network_client,
                self.VNET_NAME,
                self.SUBNET_NAME,
                self.NSG_NAME,
                self.RG_NAME,
            )
        else:
            if subnet_result is None or nsg_result is None:
                raise ValueError(
                    "Attempted to call Cluster.create with create_shared_resources=False, however "
                    f"subnet_result is {subnet_result} and nsg_result is {nsg_result}."
                    "Both are required to be populated (ie, not None)."
                )

            self.subnet = subnet_result
            self.nsg = nsg_result

        self.parallelize(
            super().create,
            create_shared_resources=False,
            subnet_result=self.subnet,
            nsg_result=self.nsg,
        )

        # We need to check that all of the VMs exist since creation sometimes fails silently
        self.check_vms_exist()

    def delete(self):
        """
        Delete the resource group for the cluster, which deletes all VMs, NICs,
        subnet, vnet, NSG, etc. associated with the cluster as well.
        """

        poller = self.resource_client.resource_groups.begin_delete(self.RG_NAME)
        poller.result()
        print(f"Deleted resource group {self.RG_NAME}")

    def setup(self):
        self.parallelize(super().setup)

    def start(self):
        self.parallelize(super().start)

    def stop(self):
        self.parallelize(super().stop)

    def selftest(self):
        self.parallelize(super().selftest)

    def wipe_mlagility_cache(self):
        self.parallelize(super().wipe_mlagility_cache)

    def run(self, input_files: str):
        # TODO: create more jobs based on permutations such as
        # batch size, data type, etc.
        device_type = hardware_to_device[self.hardware]

        jobs = [
            benchit_prefix(f"mlagility{input[2:]} --lean-cache --device {device_type}")
            for input in input_files
        ]

        print(jobs)

        job = jobs.pop(0)
        busy = True
        while job is not None or busy is True:
            job_assigned = False

            # Attempt to assign job to cluster
            if job is not None:
                for vm in self.devices:
                    if vm.async_process is None:
                        vm.run_async_command(job)
                        job_assigned = True
                        print(f"Job {job} assigned to {vm.VM_NAME}")
                        break

            # Check if any VMs are done with their last job
            for vm in self.devices:
                if vm.async_process is not None:
                    print("Polling VM", vm.VM_NAME)
                    if vm.poll_async_command():
                        vm.wipe_models_cache()

            # Get a new job if this job was assigned
            if job_assigned and len(jobs) > 0:
                print("Jobs remaining", len(jobs))
                job = jobs.pop(0)
            elif job_assigned and len(jobs) == 0:
                job = None

            # Determine whether the cluster is busy or not
            busy = False
            for vm in self.devices:
                if vm.async_process is not None:
                    busy = True

            time.sleep(1)

        result_dir = "result"

        # Gather cache directories
        for vm in self.devices:
            vm_result_dir = os.path.join(result_dir, vm.VM_NAME)
            vm.get_dir(".cache/mlagility", vm_result_dir)

        # Create report
        local_command(
            f"conda run -n mla benchit cache report -d {result_dir}/{self.name}*/.cache/mlagility -r result"
        )


def cluster_prefix(name, hardware):
    return f"{name}-{hardware}"


class SuperCluster(Cluster):
    """
    Handle for controller a cluster of clusters of VMs

    Note that most methods, including init, are inherited from Cluster. Only
    a few methods need to be overloaded to enable cluster-of-cluster support.
    """

    def populate_devices(self, retrieve: bool):
        if retrieve:
            size = None
        else:
            if self.size is None:
                raise ValueError("size must be set when retrieve=False")

            size = self.size

        self.devices = [
            Cluster(
                name=cluster_prefix(self.name, hardware),
                retrieve=retrieve,
                size=size,
                hardware=hardware,
                rg_name=self.RG_NAME,
                nsg_name=self.NSG_NAME,
                vnet_name=self.VNET_NAME,
                subnet_name=self.SUBNET_NAME,
            )
            for hardware in self.hardware
        ]

    def run(self, input_files: str):
        self.parallelize(super().run, input_files=input_files)


def main():

    start_time = time.time()

    # Define the argument parser
    parser = argparse.ArgumentParser(description="Manage MLAgility Azure VMs")

    parser.add_argument(
        "commands",
        nargs="+",
        choices=[
            "create",
            "setup",
            "start",
            "info",
            "selftest",
            "run",
            "wipe-mla-cache",
            "stop",
            "delete",
        ],
        help="Execute one or more commands on Azure. "
        "If you issue multiple commands, they will run in the order provided.",
    )

    parser.add_argument(
        "--cluster",
        "-c",
        help="Work with a cluster",
        action="store_true",
    )

    parser.add_argument(
        "--size",
        "-s",
        help="Size of VM cluster",
        type=int,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--name",
        "-n",
        help="Name that will prefix all the of resources. name-rg, name-vm, name-nic, etc.",
        required=False,
        default="mla",
    )

    parser.add_argument(
        "--hardware",
        "-d",
        help="Hardware devices for VM cluster",
        required=False,
        nargs="+",
        default=["cpu-small"],
    )

    parser.add_argument(
        "--input-files",
        "-f",
        help="Path to input files for job",
        nargs="*",
    )

    # Parse the arguments
    args = parser.parse_args()

    # If "create" is one of the commands then we will create the VM(s) from scratch
    # and then use them for the duration of the session. If "create" is not one of the
    # commands we will attempt to retrieve VM(s) that already exist.
    retrieve = "create" not in args.commands

    # Create a handle to a VM, Cluster of VMs, or SuperCluster (cluster of clusters)
    if args.cluster:
        if len(args.hardware) > 1:
            handle = SuperCluster(
                args.name,
                retrieve=retrieve,
                size=args.size,
                hardware=args.hardware,
            )
        else:
            handle = Cluster(
                args.name,
                retrieve=retrieve,
                size=args.size,
                hardware=args.hardware[0],
            )
    else:
        if len(args.hardware) > 1:
            raise ValueError(
                "Length of hardware arg must be 1 if --cluster is not used"
            )
        handle = Device(args.name, retrieve, args.hardware[0])

    command_to_function = {
        "create": handle.create,
        "setup": handle.setup,
        "start": handle.start,
        "info": handle.info,
        "selftest": handle.selftest,
        "wipe-mla-cache": handle.wipe_mlagility_cache,
        "stop": handle.stop,
        "delete": handle.delete,
    }

    for command in args.commands:
        if command == "run":
            handle.run(args.input_files)
        else:
            command_to_function[command]()

    end_time = time.time()
    total_time = end_time - start_time

    print("Time elapsed (seconds):", total_time)


if __name__ == "__main__":
    main()
