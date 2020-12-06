import os

from tfl_manager import TFLMan


def get_frames(pls_file):
    pls = []
    found_pkl = False
    pkl = ''
    with open(pls_file, 'r') as f:
        for line in f:
            if line != "":
                line = line.split("#")[0]
                line = line.strip()
                if line == "":
                    continue
                line = os.path.abspath(line)
                if line != "" and os.path.exists(line):
                    if not found_pkl and ".pkl" in line:
                        pkl = line
                        continue
                    else:
                        pls.append(line)
    return pkl, pls


def init():
    pls_file = "pls.txt"
    pkl, frame_list = get_frames(pls_file)
    tfl_manager = TFLMan(pkl, frame_list)
    return pkl, frame_list, tfl_manager


def run():
    pkl, frame_list, tfl_manager = init()
    tfl_manager.run(2)

def main():
    run()


if __name__ == '__main__':
    main()