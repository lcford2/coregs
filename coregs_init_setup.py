import requests
import tarfile
import zipfile
import os
import shutil
import glob
import subprocess

graps_version = "1.0-coregs"
temoa_version = "1.2.0-coregs"

graps_url = f"https://github.com/lcford2/GRAPS/archive/refs/tags/v{graps_version}.tar.gz"
temoa_url = f"https://github.com/lcford2/temoa/archive/refs/tags/v{temoa_version}.tar.gz"
data_url = "https://zenodo.org/record/6315941/files/coregs-input-data.zip?download=1"

graps_outfile = f"./graps-{graps_version}.tar.gz"
temoa_outfile = f"./temoa-{temoa_version}.tar.gz"
data_outfile = "./coregs-input-data.zip"

def download_file(url, outfile):
    file = requests.get(url)
    with open(outfile, "wb") as f:
        f.write(file.content)


def extract_tarball(tarball, output="."):
    with tarfile.open(tarball) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, output)


def extract_zipfile(zipfile_loc, output="."):
    try:
        with zipfile.ZipFile(zipfile_loc, "r") as f:
            f.extractall(output)
    except NotImplementedError:
        print(f"Extracting {zipfile_loc} failed. Trying `unzip` command line utility.")
        subprocess.call([
            "unzip",
            zipfile_loc,
            "-d", 
            output
        ])

def check_dir_exist(dirloc):
    return os.path.isdir(dirloc)


def check_dir_empty(dirloc):
    return len(os.listdir(dirloc)) == 0


def ask_overwrite_dir(dirloc):
    resp = input(
        f"{dirloc} is not empty. Do you wish to overwrite it? [y/N] "
    )
    return resp.lower()


def prep_dirloc(dirloc):
    if not check_dir_exist(dirloc):
        # dirloc does not exist, create it and return
        os.mkdir(dirloc)
        return True
    elif not check_dir_empty(dirloc):
        # dirloc exists and is not empty
        # ask if it should be overwritten
        overwrite = ask_overwrite_dir(dirloc) == "y"
        if overwrite:
            shutil.rmtree(dirloc)
            os.mkdir(dirloc)
            return True
        else:
            return False
    else:
        # dirloc exists and is empty so all is good
        return True

def move_dir_files(dirloc, output_dir):
    for file in glob.glob(f"{dirloc}/*"):
        if os.path.isfile(file):
            output_file = f"{output_dir}/{os.path.basename(file)}"
            if os.path.isfile(output_file):
                resp = ask_overwrite_dir(output_file)
                if resp == "y":
                    os.remove(output_file)
                    shutil.move(file, output_file)
            else:
                shutil.move(file, output_file)
    

def get_graps_temoa():
    print(f"\nDownloading GRAPS {graps_version} from {graps_url} to {graps_outfile}")
    download_file(graps_url, graps_outfile)

    print(f"Downloading Temoa {temoa_version} from {temoa_url} to {temoa_outfile}")
    download_file(temoa_url, temoa_outfile)

    print(f"\nExtracting {graps_outfile}")
    extract_tarball(graps_outfile)

    print(f"Extracting {temoa_outfile}")
    extract_tarball(temoa_outfile)

    print("\nMoving GRAPS and Temoa to proper locations for COREGS")
    graps_dir = "./graps"
    temoa_dir = "../temoa"

    graps_ready = prep_dirloc(graps_dir)
    if graps_ready:
        os.rename(f"GRAPS-{graps_version}", graps_dir)
    else:
        print(f"Not moving files from GRAPS-{graps_version} to {graps_dir}")
        

    temoa_ready = prep_dirloc(temoa_dir)
    if temoa_ready:
        os.rename(f"temoa-{temoa_version}", temoa_dir)
    else:
        print(f"Not moving files from temoa-{temoa_version} to {temoa_dir}")

    print("\nCleaning up")
    os.remove(graps_outfile)
    os.remove(temoa_outfile)


def get_coregs_data():
    print(f"\nDownloading COREGS input data from {data_url} to {data_outfile}")
    download_file(data_url, data_outfile)

    print(f"\nExtracting {data_outfile}")
    extract_zipfile(data_outfile)

    print("\nMoving data to proper location")
    data_dir = "./data"

    # if the dir does not exist, make it
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # if is exists but is not a directory, fail
    elif not os.path.isdir(data_dir):
        raise NotADirectoryError(F"ERROR: Cannot move data to {data_dir} because it is a file.")
    
    move_dir_files("./coregs-input-data", data_dir)
        
    print("\nCleaning up")
    os.remove(data_outfile)
    shutil.rmtree("./coregs-input-data")
    

def main():
    get_graps_temoa()
    get_coregs_data()


if __name__ == "__main__":
    main()
