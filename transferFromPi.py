import paramiko
from scp import SCPClient
import time
import os

# Define the remote and local paths
remote_path = '~/Desktop/camera/image.jpg'
local_dir = os.path.expanduser('~/Desktop/presloh_camera/photos/')

# Ensure the local directory exists
os.makedirs(local_dir, exist_ok=True)

# Define the connection parameters
hostname = '10.1.10.50'
port = 3456
username = 'presloh'
# If using a private key for authentication, specify the path to the private key
# private_key_path = os.path.expanduser('~/.ssh/id_rsa')

def create_scp_client():
    # Create an SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # Connect to the remote server
    ssh.connect(hostname, port=port, username=username)
    # Create an SCP client
    scp = SCPClient(ssh.get_transport())
    return scp

def transfer_image(scp):
    try:
        # Get the current epoch time
        epoch_time = int(time.time())
        # Define the local file path with the epoch time appended to the file name
        local_path = os.path.join(local_dir, f'image_{epoch_time}.jpg')
        # Transfer the file from the remote server to the local machine
        scp.get(remote_path, local_path)
        print(f'Transferred image to {local_path}')
    except Exception as e:
        print(f'Error transferring image: {e}')

def main():
    while True:
        try:
            # Create an SCP client and transfer the image
            scp = create_scp_client()
            transfer_image(scp)
            # Close the SCP client
            scp.close()
        except Exception as e:
            print(f'Error in SCP connection: {e}')
        # Wait for 40 seconds before the next transfer
        time.sleep(40)

if __name__ == '__main__':
    main()
