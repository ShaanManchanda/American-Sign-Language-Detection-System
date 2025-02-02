import copy
import itertools#functions that create iterators for efficient looping

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x#centering it relative to the first point
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y#centering it relative to the first point

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))#Flattens the list of landmark coordinates

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list