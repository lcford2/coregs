import requests
import tarfile
import zipfile
import os
import shutil
import glob
import subprocess
import argparse

from coregs_config import graps_loc, temoa_loc

graps_version = "1.0.1-coregs"
temoa_version = "1.2.0-coregs"

graps_url = f"https://github.com/lcford2/GRAPS/archive/refs/tags/v{graps_version}.tar.gz"
temoa_url = f"https://github.com/lcford2/temoa/archive/refs/tags/v{temoa_version}.tar.gz"
data_url = "https://zenodo.org/record/6315941/files/coregs-input-data.zip?download=1"

graps_outfile = f"./graps-{graps_version}.tar.gz"
temoa_outfile = f"./temoa-{temoa_version}.tar.gz"
data_outfile = "./coregs-input-data.zip"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Helper script to download and install GRAPS and Temoa"
                    " and to download the TVA data from Zenodo. If no arguments"
                    " are provided, GRAPS, Temoa, and the data will all be"
                    " downloaded. If any flags for a specific component is"
                    " included, only the components specifically indicated"
                    " are downloaded."
    )
    parser.add_argument(
        "-G",
        "--graps",
        action="store_true",
        help="Download and install GRAPS in correct location"
    )
    parser.add_argument(
        "-T",
        "--temoa",
        action="store_true",
        help="Download and install Temoa in correct location"
    )
    parser.add_argument(
        "-D",
        "--data",
        action="store_true",
        help="Download and place TVA data in correct location"
    )
    args = parser.parse_args()

    if not any([args.graps, args.temoa, args.data]):
        args.graps = True
        args.temoa = True
        args.data  = True
    return args


def download_file(url, outfile):
    file = requests.get(url)
    with open(outfile, "wb") as f:
        f.write(file.content)


def extract_tarball(tarball, output="."):
    with tarfile.open(tarball) as f:
        f.extractall(output)


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


def get_model(model, version, url, outfile):
    print(f"\nDownloading {model.capitalize()} {version} from {url} to {outfile}")
    download_file(url, outfile)

    print(f"\nExtracting {outfile}")
    extract_tarball(outfile)
    
    if model == "GRAPS":
        outdir = graps_loc
    elif model == "temoa":
        outdir = temoa_loc

    print(f"\nMoving {model.capitalize()} to {outdir} for COREGS")

    loc_ready = prep_dirloc(outdir)
    if loc_ready:
        os.rename(f"{model}-{version}", outdir)
    else:
        print(f"Not moving files from {model}-{version} to {outdir}")

    print("\nCleaning up")
    os.remove(outfile)


def get_graps():
    get_model("GRAPS", graps_version, graps_url, graps_outfile)


def get_temoa():
    get_model("temoa", temoa_version, temoa_url, temoa_outfile)


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
    args = parse_args()
    if args.graps:
        get_graps()
    
    if args.temoa:
        get_temoa()

    if args.data:
        get_coregs_data()


if __name__ == "__main__":
    main()
