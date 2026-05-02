import argparse
import feature_extractor
import numpy as np
import os


def split_data(filename, outdir):
    txt_file = open(filename)
    count = 0
    feature_vecs = []
    for line in txt_file:
        count += 1
        new_file_name = outdir + '/' + str(count)+'.txt'
        write_file = open(new_file_name, "w+")
        write_file.write(line)
        write_file.close()
        print(new_file_name)

        vector_list = feature_extractor.process_file(new_file_name)
        feature_vecs.append(vector_list)
        os.remove(new_file_name)

    feature_vec_array = np.array(feature_vecs)
    np.save(outdir+'_features.npy', feature_vec_array)


def parse_command_line():
    description = 'data splitter'
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('filepath', metavar='filepath', type=str,
                           help='Filepath of json data to split')
    argparser.add_argument('outdir', metavar='outdir', type=str,
                           help='Filepath of output directory')
    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_command_line()
    split_data(args.filepath, args.outdir)
