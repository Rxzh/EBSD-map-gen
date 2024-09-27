import os
def rename_files_in_folder(folder):
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith('_VAL.npy'):
                new_name = filename.replace('_VAL.npy', '.npy')
            elif filename.endswith('_TRAIN.npy'):
                new_name = filename.replace('_TRAIN.npy', '.npy')
            elif filename.endswith('_TEST.npy'):
                new_name = filename.replace('_TEST.npy', '.npy')
            else:
                continue

            old_file = os.path.join(dirpath, filename)
            new_file = os.path.join(dirpath, new_name)
            os.rename(old_file, new_file)


if __name__ == '__main__':
    data_folder = 'data'
    rename_files_in_folder(data_folder)
