import glob
import os


import os
import glob

def labels_to_number(path):
    """
    Convert string labels in numbers.

    :param path: path to folder.
    :return: a dictionary.
    """
    # Add a slash at the end if it's missing to ensure correct directory globbing
    if not path.endswith(os.path.sep):
        path += os.path.sep
        
    # Use '*/' to ensure glob only matches directories
    directories = glob.glob(path + '*/')
    print("Directories found:", directories)
    
    classes = [os.path.basename(os.path.normpath(i)) for i in directories]
    classes.sort()
    print("Sorted classes:", classes)

    labels_dict = {label: i for i, label in enumerate(classes)}
    print("Labels dictionary:", labels_dict)

    return labels_dict



def videos_to_dict(path, labels):
    """
    Read the videos and return a dict like {'path_to_video', 'label'}.

    :param path: path to videos folder.
    :param labels: labels as dict.
    :return: a dictionary.
    """
    videos_dict = {}
    for root, dirs, files in os.walk(os.path.relpath(path)):
        for file in files:
            video_name = os.path.join(root, file)
            dir_name = os.path.basename(os.path.dirname(video_name))  # label
            if dir_name in labels:
                videos_dict[video_name] = labels[dir_name]
            else:
                print(f"Warning: No label found for directory '{dir_name}'. Skipping file '{file}'")
    return videos_dict

