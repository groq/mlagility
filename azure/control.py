import os
import argparse
import subprocess
import shutil
import time
from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient
from azure.mgmt.resource import ResourceManagementClient

LOCATION = "eastus"
RESOURCE_GROUP = "mla-resource"
VM_NAME = "ExampleVM"
USERNAME = "azureuser"
PASSWORD = "donthackmla"
SSH_PATH = "~/.ssh/mla_key.pub"
SSH_PRIVATE_PATH = "~/.ssh/mla_key.pem"

device_to_image_reference = {
    "nvidia": {
        "publisher": "Nvidia",
        "offer": "pytorch_from_nvidia",
        "sku": "ngc-pytorch-version-22-10-0_gen2",
        "version": "22.10.0",
    },
    "cpu": {
        "publisher": "canonical",
        "offer": "0001-com-ubuntu-server-focal",
        "sku": "20_04-lts-gen2",
        "version": "latest",
    },
}

hardware_to_device = {
    "t4": "nvidia",
    "icelake": "cpu",
    "cpu-small": "cpu",
}

hardware_to_vm_size = {
    "t4": "Standard_NC4as_T4_v3",  # Nvidia T4 with 4 EPYC vCPUs
    "icelake": "Standard_D16ds_v5",  # 16-core Xeon IceLake
    "cpu-small": "Standard_E2s_v3",  # 2-core Xeon (unknown generation)
}

vm_size_to_hardware = {v: k for k, v in hardware_to_vm_size.items()}

device_to_plan = {
    "nvidia": {
        "name": "ngc-pytorch-version-22-10-0_gen2",
        "product": "pytorch_from_nvidia",
        "publisher": "nvidia",
    },
    "cpu": None,
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


def create_network_security_group(
    network_client: NetworkManagementClient, name_prefix: str
):
    nsg_name = f"{name_prefix}-nsg"

    # Create Security Rules
    # NOTE: These are a copy of the default security rules every VM gets
    # in Azure portal. May want to customize eventually.
    poller = network_client.security_rules.begin_create_or_update(
        RESOURCE_GROUP,
        nsg_name,
        f"{name_prefix}-JupyterHub",
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
        RESOURCE_GROUP,
        nsg_name,
        f"{name_prefix}-RStudio_Server",
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
        RESOURCE_GROUP,
        nsg_name,
        f"{name_prefix}-SSH",
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

    # Create security group

    poller = network_client.network_security_groups.begin_create_or_update(
        RESOURCE_GROUP,
        nsg_name,
        parameters={
            "securityRules": [
                jupyter_rule,
                rstudio_rule,
                ssh_rule,
            ]
        },
    )

    nsg = poller.result()

    return nsg


def create_vnet_and_subnet(
    network_client: NetworkManagementClient, vnet_name, subnet_name
):
    # Step 2: provision a virtual network

    # A virtual machine requires a network interface client (NIC). A NIC
    # requires a virtual network and subnet along with an IP address.
    # Therefore we must provision these downstream components first, then
    # provision the NIC, after which we can provision the VM.

    poller = network_client.virtual_networks.begin_create_or_update(
        RESOURCE_GROUP,
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
        RESOURCE_GROUP,
        vnet_name,
        subnet_name,
        {"address_prefix": "10.0.0.0/24"},
    )
    subnet_result = poller.result()

    print(
        f"Provisioned virtual subnet {subnet_result.name} with address \
    prefix {subnet_result.address_prefix}"
    )

    return vnet_result, subnet_result


class VM:
    """
    Handle for controlling an individual VM
    """

    def __init__(self, name, retrieve: bool = True, hardware=None):
        self.VM_NAME = name
        self.VNET_NAME = f"{name}-vnet"
        self.SUBNET_NAME = f"{name}-subnet"
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

        if retrieve:
            vm_size, _ = self.info(verbosity="none")
            self.hardware = vm_size_to_hardware[vm_size]
        else:
            if hardware is None:
                raise ValueError("hardware must be specified when creating VM")

            self.hardware = hardware

        self.device = hardware_to_device[self.hardware]

    @property
    def ip_address(self):
        if not self._ip:
            self._ip = self.network_client.public_ip_addresses.get(
                RESOURCE_GROUP, self.IP_NAME
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
            RESOURCE_GROUP, self.VM_NAME, expand="instanceView"
        )

        status = vm_info.instance_view.statuses[1].display_status
        size = vm_info.hardware_profile.vm_size

        if verbosity == "high":
            print(f"{self.VM_NAME} ({size}, {self.ip_address}): {status}")

        return size, status

    def create(self):
        group_list = self.resource_client.resource_groups.list()
        resource_group_names = [item.name for item in group_list]
        assert RESOURCE_GROUP in resource_group_names

        vnet_result, subnet_result = create_vnet_and_subnet(
            self.network_client, self.VNET_NAME, self.SUBNET_NAME
        )

        # Step 4: Provision an IP address and wait for completion
        poller = self.network_client.public_ip_addresses.begin_create_or_update(
            RESOURCE_GROUP,
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
            RESOURCE_GROUP,
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
                "network_security_group": {
                    "id": "/subscriptions/f9f04fac-965a-4f5e-b4b8-a3ec406a9047/resourceGroups/mla-resource/providers/Microsoft.Network/networkSecurityGroups/mlacentralnsg718"
                },
            },
        )

        nic_result = poller.result()

        with open(os.path.expanduser(SSH_PATH)) as keyfile:
            ssh_key = keyfile.read()

        vm_spec = {
            "location": LOCATION,
            "storage_profile": {
                "image_reference": device_to_image_reference[self.device],
                "os_disk": {
                    "disk_size_gb": 64,
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
            RESOURCE_GROUP,
            self.VM_NAME,
            vm_spec,
        )

        vm_result = poller.result()

        print(f"Provisioned virtual machine {vm_result.name}")

    def delete(self):
        print("Deleting VM", self.VM_NAME)
        # NOTE: we call result() to make sure the delete operation finishes before we move on to the next one
        self.compute_client.virtual_machines.begin_delete(
            resource_group_name=RESOURCE_GROUP, vm_name=self.VM_NAME
        ).result()

        print("Deleting nic", self.NIC_NAME)
        self.network_client.network_interfaces.begin_delete(
            RESOURCE_GROUP, self.NIC_NAME
        ).result()
        print("Deleting ip", self.IP_NAME)
        self.network_client.public_ip_addresses.begin_delete(
            RESOURCE_GROUP, self.IP_NAME
        ).result()
        print("Deleting subnet", self.SUBNET_NAME)
        self.network_client.subnets.begin_delete(
            RESOURCE_GROUP, self.VNET_NAME, self.SUBNET_NAME
        ).result()
        print("Deleting vnet", self.VNET_NAME)
        self.network_client.virtual_networks.begin_delete(
            RESOURCE_GROUP, self.VNET_NAME
        ).result()

    def stop(self):
        print("Stopping VM", self.VM_NAME)
        self.compute_client.virtual_machines.begin_deallocate(
            RESOURCE_GROUP, self.VM_NAME
        )

    def start(self):
        print("Starting VM", self.VM_NAME)
        self.compute_client.virtual_machines.begin_start(RESOURCE_GROUP, self.VM_NAME)

    def send_file(self, file):
        command = f"scp -i {SSH_PRIVATE_PATH} {file} {USERNAME}@{self.ip_address}:~/."
        print(f"Running command on {self.VM_NAME}: {command}")
        subprocess.run(command.split(" "), check=True)

    def get_file(self, remote_file, local_dir, local_file_name):
        os.makedirs(local_dir, exist_ok=True)
        command = f"scp -i {SSH_PRIVATE_PATH} {USERNAME}@{self.ip_address}:{remote_file} {local_dir}/{local_file_name}"
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
        command = f"scp -i {SSH_PRIVATE_PATH} {USERNAME}@{self.ip_address}:{cache_tar_filename} {local_destination}"
        print(f"Running command on {self.VM_NAME}: {command}")
        subprocess.run(command.split(" "), check=True)

        # Untar
        local_cache_tar_path = os.path.join(local_destination, cache_tar_filename)
        local_command(f"tar -xzvf {local_cache_tar_path} -C {local_destination}")

        # Clean up the tar file
        local_command(f"rm {local_cache_tar_path}")

    def run_command(self, command):
        full_command = (
            f"ssh -i {SSH_PRIVATE_PATH} {USERNAME}@{self.ip_address} {command}"
        )
        print(f"Running command on {self.VM_NAME}: {full_command}")
        subprocess.run(full_command.split(" "), check=True)

    def check_command(self, command, success_term):
        full_command = (
            f"ssh -i {SSH_PRIVATE_PATH} {USERNAME}@{self.ip_address} {command}"
        )
        print(f"Running command on {self.VM_NAME}: {full_command}")
        result = subprocess.check_output(full_command.split(" ")).decode()
        if success_term in result:
            print(f"\n\nSuccess on {self.VM_NAME}!\n\n")
        else:
            raise Exception("Error! Success term not in result")

    def run_async_command(self, command):
        full_command = (
            f"ssh -i {SSH_PRIVATE_PATH} {USERNAME}@{self.ip_address} {command}"
        )
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

    def finish_aysnc_command(self):
        """
        Waits for an async command to finish and then prints the result
        """
        out, errs = self.async_process.communicate()
        print(f"{self.VM_NAME} finished job: {self.async_command}")
        print(out.decode())
        print(errs.decode())

        self.async_process = None

        return out, errs

    def begin_setup(self):
        # Add the VM IP to known hosts, and retry a few times before giving up
        known_host_success = False
        timeout = 300  # seconds
        sleep_amount = 15  # seconds
        assert timeout % sleep_amount == 0

        while not known_host_success and timeout > 0:
            try:
                self.add_to_known_hosts()
                known_host_success = True
            except subprocess.CalledProcessError as e:
                timeout = timeout - sleep_amount
                if timeout > 0:
                    print(
                        "Adding to known hosts failed, which means the VM is probably "
                        f"not ready yet. Re-attempting, timeout remaining is {timeout}"
                    )
                else:
                    print(
                        "Adding to known hosts failed and the timeout has elapsed. Exiting."
                    )
                    raise e
                time.sleep(sleep_amount)

        self.send_file("setup_part_1.sh")
        self.send_file("setup_part_2.sh")
        self.run_async_command("bash setup_part_1.sh")

    def setup_part_2(self):
        print(f"\n\nSETUP PART 1 RESULT FOR {self.VM_NAME}:\n\n")
        self.finish_aysnc_command()

        self.run_async_command("bash setup_part_2.sh")

    def finish_setup(self):
        print(f"\n\nSETUP FINAL RESULT FOR {self.VM_NAME}:\n\n")
        self.finish_aysnc_command()

    def setup(self):
        self.begin_setup()
        self.setup_part_2()
        self.finish_setup()

    def selftest(self):
        self.check_command(
            benchit_prefix("mlagility/models/selftest/linear.py"),
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


def vm_prefix(name: str) -> str:
    return f"{name}-vm-"


class Cluster:
    """
    Handle for controlling a cluster of VMs
    """

    def __init__(
        self,
        name: str,
        retrieve: bool,
        size: int = None,
        hardware: str = "cpu-small",
    ):
        self.size = size
        self.name = name
        self.hardware = hardware

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

        self.vms = None

        self.populate_vms(retrieve)

    def populate_vms(self, retrieve: bool):
        if retrieve:
            resource_group_vms = self.compute_client.virtual_machines.list(
                RESOURCE_GROUP
            )
            self.vms = [
                VM(vm.name, retrieve=True)
                for vm in resource_group_vms
                if vm.name.startswith(vm_prefix(self.name))
            ]
        else:
            if self.size is None:
                raise ValueError("size must be set when retrieve=False")

            self.vms = [
                VM(
                    name=f"{vm_prefix(self.name)}{i}",
                    retrieve=False,
                    hardware=self.hardware,
                )
                for i in range(self.size)
            ]

    def info(self):
        for vm in self.vms:
            vm.info()

    def create(self):
        for vm in self.vms:
            vm.create()

    def delete(self):
        for vm in self.vms:
            vm.delete()

    def begin_setup(self):
        for vm in self.vms:
            vm.begin_setup()

    def setup_part_2(self):
        for vm in self.vms:
            vm.setup_part_2()

    def finish_setup(self):
        for vm in self.vms:
            vm.finish_setup()

    def setup(self):
        self.begin_setup()
        self.setup_part_2()
        self.finish_setup()

    def start(self):
        for vm in self.vms:
            vm.start()

    def stop(self):
        for vm in self.vms:
            vm.stop()

    def selftest(self):
        for vm in self.vms:
            vm.selftest()

    def run_async_command(self, command):
        """
        Run the same async command on all VMs in the cluster, then
        wait for all VMs to finish
        """

        for vm in self.vms:
            vm.run_async_command(command)

        for vm in self.vms:
            vm.finish_aysnc_command()

    def wipe_mlagility_cache(self):
        for vm in self.vms:
            vm.wipe_mlagility_cache()

    def run(self, input_files: str):
        # TODO: create more jobs based on permutations such as
        # hardware type, batch size, data type, etc.
        jobs = [
            benchit_prefix(f"mlagility{input[2:]} --lean-cache")
            for input in input_files
        ]

        print(jobs)

        job = jobs.pop(0)
        busy = True
        while job is not None or busy == True:
            job_assigned = False

            # Attempt to assign job to cluster
            if job is not None:
                for vm in self.vms:
                    if vm.async_process is None:
                        vm.run_async_command(job)
                        job_assigned = True
                        print(f"Job {job} assigned to {vm.VM_NAME}")
                        break

            # Check if any VMs are done with their last job
            for vm in self.vms:
                if vm.async_process is not None:
                    print("Polling VM", vm.VM_NAME)
                    if vm.poll_async_command():
                        vm.wipe_models_cache()

                    time.sleep(1)

            # Get a new job if this job was assigned
            if job_assigned and len(jobs) > 0:
                job = jobs.pop(0)
            elif job_assigned and len(jobs) == 0:
                job = None

            # Determine whether the cluster is busy or not
            busy = False
            for vm in self.vms:
                if vm.async_process is not None:
                    busy = True

        result_dir = "result"

        # Gather cache directories
        for vm in self.vms:
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

    def populate_vms(self, retrieve: bool):
        if retrieve:
            self.vms = [
                Cluster(
                    name=cluster_prefix(self.name, hardware),
                    retrieve=True,
                    size=None,
                    hardware=hardware,
                )
                for hardware in self.hardware
            ]

        else:
            if self.size is None:
                raise ValueError("size must be set when retrieve=False")

            self.vms = [
                Cluster(
                    name=cluster_prefix(self.name, hardware),
                    retrieve=False,
                    size=self.size,
                    hardware=hardware,
                )
                for hardware in self.hardware
            ]

    def run_async_command(self, command):
        """
        Run the same async command on all VMs in the cluster, then
        wait for all VMs to finish
        """

        for cluster in self.vms:
            for vm in cluster:
                vm.run_async_command(command)

        for cluster in self.vms:
            for vm in cluster:
                vm.finish_aysnc_command()


def main():
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
        "--get",
        "-g",
        help="Retrieve existing VM(s)",
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
        help="Name of VM/cluster",
        required=False,
        default="mla-cluster",
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

    # Create a handle to a VM, Cluster of VMs, or SuperCluster (cluster of clusters)
    if args.cluster:
        if len(args.hardware) > 1:
            if args.get:
                handle = SuperCluster(
                    args.name,
                    retrieve=True,
                    size=None,
                    hardware=args.hardware,
                )
            else:
                handle = SuperCluster(
                    args.name,
                    retrieve=False,
                    size=args.size,
                    hardware=args.hardware,
                )
        else:
            if args.get:
                handle = Cluster(
                    args.name,
                    retrieve=True,
                    size=None,
                    hardware=args.hardware[0],
                )
            else:
                handle = Cluster(
                    args.name,
                    retrieve=False,
                    size=args.size,
                    hardware=args.hardware[0],
                )
    else:
        if len(args.hardware) > 1:
            raise ValueError(
                "Length of hardware arg must be 1 if --cluster is not used"
            )
        if args.get:
            handle = VM(args.name, retrieve=True)
        else:
            handle = VM(args.name, False, args.hardware[0])

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
            # Special case because it takes an argument
            handle.run(args.input_files)
        else:
            command_to_function[command]()

    if args.run:
        handle.run(args.input_files)


if __name__ == "__main__":
    main()
