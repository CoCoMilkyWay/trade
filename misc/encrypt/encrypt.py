import os
import hashlib
import shutil
from Crypto.Cipher import AES
from Crypto import Random
import zlib
from tqdm import tqdm

# Function to generate a random key
def generate_key(password):
    key = hashlib.sha256(password.encode()).digest()
    return key

# Function to pad the data to be encrypted
def pad(s):
    return s + b"\0" * (AES.block_size - len(s) % AES.block_size)

# Function to encrypt a file
def encrypt_file(key, filename):
    chunksize = 64 * 1024
    output_filename = filename + '.enc'
    iv = Random.new().read(AES.block_size)
    
    encryptor = AES.new(key, AES.MODE_CBC, iv)
    file_size = os.path.getsize(filename)
    
    with open(filename, 'rb') as infile:
        with open(output_filename, 'wb') as outfile:
            outfile.write(iv)
            with tqdm(total=file_size, desc=f'Encrypting {filename}', unit='B', unit_scale=True) as pbar:
                while True:
                    chunk = infile.read(chunksize)
                    if len(chunk) == 0:
                        break
                    elif len(chunk) % AES.block_size != 0:
                        chunk = pad(chunk)
                    outfile.write(encryptor.encrypt(chunk))
                    pbar.update(len(chunk))
    return output_filename

# Function to decrypt a file
def decrypt_file(key, filename):
    chunksize = 64 * 1024
    output_filename = os.path.splitext(filename)[0]
    file_size = os.path.getsize(filename)
    
    with open(filename, 'rb') as infile:
        iv = infile.read(AES.block_size)
        decryptor = AES.new(key, AES.MODE_CBC, iv)
        
        with open(output_filename, 'wb') as outfile:
            with tqdm(total=file_size, desc=f'Decrypting {filename}', unit='B', unit_scale=True) as pbar:
                while True:
                    chunk = infile.read(chunksize)
                    if len(chunk) == 0:
                        break
                    outfile.write(decryptor.decrypt(chunk))
                    pbar.update(len(chunk))
    return output_filename

# Function to compress a folder
def compress_folder(folder_path):
    compressed_folder_name = os.path.basename(folder_path) + '_compressed'
    compressed_folder_path = os.path.join(os.path.dirname(folder_path), compressed_folder_name)
    shutil.make_archive(compressed_folder_path, 'zip', folder_path)
    return compressed_folder_path + '.zip'

# Function to decompress a folder
def decompress_folder(zip_file):
    target_folder_path = os.path.splitext(zip_file)[0]
    shutil.unpack_archive(zip_file, target_folder_path, 'zip')
    return target_folder_path

# Encrypting a folder
def encrypt_folder(folder_path, password):
    key = generate_key(password)
    compressed_folder = compress_folder(folder_path)
    encrypted_zip = encrypt_file(key, compressed_folder)
    os.remove(compressed_folder)
    return encrypted_zip

# Decrypting a folder
def decrypt_folder(folder_path, password):
    key = generate_key(password)
    encrypted_zip = None
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.enc'):
                file_path = os.path.join(root, file)
                decrypted_zip = decrypt_file(key, file_path)
                encrypted_zip = file_path
    if encrypted_zip:
        os.remove(encrypted_zip)

# Example usage
if __name__ == "__main__":
    folder_to_encrypt = './audio'
    password = 'your_password'
    
    print(f"Password: {password}")
    
    # Encrypt the folder
    encrypted_zip = encrypt_folder(folder_to_encrypt, password)
    
    # Decrypt the folder
    decrypt_folder(os.path.dirname(encrypted_zip), password)
    
    # Automatically unzip the decrypted folder
    decompressed_folder = decompress_folder(os.path.splitext(encrypted_zip)[0])
    print(f"Decompressed folder: {decompressed_folder}")
