import requests
import tarfile
import os
import shutil

graps_version = "1.0-coregs"
temoa_version = "1.2.0-coregs"

graps_url = f"https://github.com/lcford2/GRAPS/archive/refs/tags/v{graps_version}.tar.gz"
temoa_url = f"https://github.com/lcford2/temoa/archive/refs/tags/v{temoa_version}.tar.gz"

graps_outfile = f"./graps-{graps_version}.tar.gz"
temoa_outfile = f"./temoa-{temoa_version}.tar.gz"

def download_file(url, outfile):
    file = requests.get(url)
    with open(outfile, "wb") as f:
        f.write(file.content)


def extract_tarball(tarball, output="."):
    with tarfile.open(tarball) as f:
        f.extractall(output)


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
            return True
        else:
            return False
    else:
        # dirloc exists and is empty so all is good
        return True


def main():
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


if __name__ == "__main__":
    main()