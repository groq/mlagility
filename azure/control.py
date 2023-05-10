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


class VM:
    """
    Handle for controlling an individual VM
    """

    def __init__(self, name):
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

    def info(self):
        status = (
            self.compute_client.virtual_machines.get(
                RESOURCE_GROUP, self.VM_NAME, expand="instanceView"
            )
            .instance_view.statuses[1]
            .display_status
        )
        print(f"{self.VM_NAME} ({self.ip_address}): {status}")

    def create(self):
        group_list = self.resource_client.resource_groups.list()
        resource_group_names = [item.name for item in group_list]
        assert RESOURCE_GROUP in resource_group_names

        # Step 2: provision a virtual network

        # A virtual machine requires a network interface client (NIC). A NIC
        # requires a virtual network and subnet along with an IP address.
        # Therefore we must provision these downstream components first, then
        # provision the NIC, after which we can provision the VM.

        poller = self.network_client.virtual_networks.begin_create_or_update(
            RESOURCE_GROUP,
            self.VNET_NAME,
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
        poller = self.network_client.subnets.begin_create_or_update(
            RESOURCE_GROUP,
            self.VNET_NAME,
            self.SUBNET_NAME,
            {"address_prefix": "10.0.0.0/24"},
        )
        subnet_result = poller.result()

        print(
            f"Provisioned virtual subnet {subnet_result.name} with address \
        prefix {subnet_result.address_prefix}"
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

        poller = self.compute_client.virtual_machines.begin_create_or_update(
            RESOURCE_GROUP,
            self.VM_NAME,
            {
                "location": LOCATION,
                "storage_profile": {
                    "image_reference": {
                        "publisher": "Canonical",
                        "offer": "0001-com-ubuntu-server-jammy",
                        "sku": "22_04-lts-gen2",
                        "version": "latest",
                    }
                },
                "hardware_profile": {"vm_size": "Standard_E2s_v3"},
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
            },
        )

        vm_result = poller.result()

        print(f"Provisioned virtual machine {vm_result.name}")

        # Give the VM a little time to open up to SSH
        time.sleep(15)
        self.add_to_known_hosts()

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
        shutil.rmtree(local_destination)

        os.makedirs(local_destination)
        command = f"scp -r -i {SSH_PRIVATE_PATH} {USERNAME}@{self.ip_address}:{remote_dir} {local_destination}"
        print(f"Running command on {self.VM_NAME}: {command}")
        subprocess.run(command.split(" "), check=True)

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
            print("Error! Success term not in result", result)

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

    def hello_world(self):
        self.check_command(
            benchit_prefix("mlagility/models/selftest/linear.py"),
            "Successfully benchmarked",
        )
        self.check_command(
            benchit_prefix("cache clean --all"), "Removed the build artifacts"
        )

    def wipe_models_cache(self):
        # NOTE: this method needs to be kept up-to-date with MLAgility's corpus
        # such that it deletes all cached model files on disk. Otherwise the
        # VM disks will fill up.
        command = "rm -rf .cache/huggingface .cache/torch-hub"
        self.run_command(command)

    def wipe_mlagility_cache(self):
        command = benchit_prefix("cache delete --all")
        self.run_command(command)


class Cluster:
    """
    Handle for controlling a cluster of VMs
    """

    def __init__(self, name: str, retrieve: bool, size: int = None):
        self.size = size
        self.name = name
        self.vm_prefix = f"{self.name}-vm-"

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

        if retrieve:
            resource_group_vms = self.compute_client.virtual_machines.list(
                RESOURCE_GROUP
            )
            self.vms = [
                VM(vm.name)
                for vm in resource_group_vms
                if vm.name.startswith(self.vm_prefix)
            ]
        else:
            if size is None:
                raise ValueError("size must be set when retrieve=False")

            self.vms = [VM(f"{self.vm_prefix}{i}") for i in range(size)]

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

    def hello_world(self):
        for vm in self.vms:
            vm.hello_world()

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


class Job:
    def __init__(self, input_files: str):
        self.input_files = input_files
        # TODO: create more jobs based on permutations such as
        # device type, batch size, data type, etc.
        self.jobs = [
            benchit_prefix(f"mlagility{input[2:]} --lean-cache")
            for input in self.input_files
        ]

        print(self.jobs)

    def run(self, cluster: Cluster):
        job = self.jobs.pop(0)
        busy = True
        while job is not None or busy == True:
            job_assigned = False

            # Attempt to assign job to cluster
            if job is not None:
                for vm in cluster.vms:
                    if vm.async_process is None:
                        vm.run_async_command(job)
                        job_assigned = True
                        print(f"Job {job} assigned to {vm.VM_NAME}")
                        break

            # Check if any VMs are done with their last job
            for vm in cluster.vms:
                if vm.async_process is not None:
                    print("Polling VM", vm.VM_NAME)
                    if vm.poll_async_command():
                        vm.wipe_models_cache()

                    time.sleep(1)

            # Get a new job if this job was assigned
            if job_assigned and len(self.jobs) > 0:
                job = self.jobs.pop(0)
            elif job_assigned and len(self.jobs) == 0:
                job = None

            # Determine whether the cluster is busy or not
            busy = False
            for vm in cluster.vms:
                if vm.async_process is not None:
                    busy = True

        # Gather cache directories
        for vm in cluster.vms:
            vm.get_dir(".cache/mlagility", vm.VM_NAME)

        # Create report
        local_command(
            f"conda run -n mla benchit cache report -d {cluster.name}*/mlagility"
        )


def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Manage MLAgility Azure VMs")

    parser.add_argument("--get", help="Retrieve existing VM(s)", action="store_true")

    # Add the arguments
    parser.add_argument("--create", help="Create VM(s)", action="store_true")

    parser.add_argument("--info", help="Get info about VM(s)", action="store_true")

    parser.add_argument("--delete", help="Delete VM(s)", action="store_true")

    parser.add_argument("--stop", help="Stop VM(s)", action="store_true")

    parser.add_argument("--start", help="Start VM(s)", action="store_true")

    parser.add_argument("--setup", help="Set up VM(s)", action="store_true")

    parser.add_argument("--cluster", help="Work with a cluster", action="store_true")

    parser.add_argument(
        "--size", help="Size of VM cluster", type=int, required=False, default=None
    )

    parser.add_argument(
        "--hello-world", help="Run hello world on the VM(s)", action="store_true"
    )

    parser.add_argument("--run", help="Run a job", action="store_true")

    parser.add_argument(
        "--input-files",
        "-f",
        help="Path to input files for job",
        nargs="*",
    )

    parser.add_argument(
        "--wipe-mla-cache", help="Wipe the mlagility cache", action="store_true"
    )

    # Parse the arguments
    args = parser.parse_args()

    if args.cluster:
        if args.get:
            handle = Cluster("mla-cluster", True)
        else:
            handle = Cluster("mla-cluster", False, args.size)
    else:
        handle = VM("ExampleVM")

    if args.info:
        handle.info()

    if args.create:
        handle.create()

    if args.delete:
        handle.delete()

    if args.setup:
        handle.setup()

    if args.stop:
        handle.stop()

    if args.start:
        handle.start()

    if args.hello_world:
        handle.hello_world()

    if args.run:
        job = Job(args.input_files)
        job.run(handle)

    if args.wipe_mla_cache:
        handle.wipe_mlagility_cache()


if __name__ == "__main__":
    main()
