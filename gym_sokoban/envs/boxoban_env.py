from .sokoban_env import SokobanEnv
from .render_utils import room_to_rgb
import os
from os import listdir
from os.path import isfile, join
import requests
import zipfile
from tqdm import tqdm
import random
import numpy as np

class BoxobanEnv(SokobanEnv):
    # These are fixed because they come from the data files
    num_boxes = 4
    dim_room = (10, 10)

    def __init__(
        self,
        max_steps=120,
        difficulty="unfiltered",
        split="train",
        cache_path=".sokoban_cache",
        render_mode="rgb_array",
        tinyworld_obs=False,
        tinyworld_render=False,
        terminate_on_first_box=False,
    ):
        self.difficulty = difficulty
        self.split = split
        self.verbose = False
        self.cache_path = cache_path
        super(BoxobanEnv, self).__init__(
            dim_room=self.dim_room,
            max_steps=max_steps,
            num_boxes=self.num_boxes,
            render_mode=render_mode,
            tinyworld_obs=tinyworld_obs,
            tinyworld_render=tinyworld_render,
            terminate_on_first_box=terminate_on_first_box,
        )
        

    def reset(self, options={}, seed=None):
        if self.difficulty == 'hard':
            # Hard has no splits
            self.train_data_dir = os.path.join(self.cache_path, 'boxoban-levels-master', self.difficulty)
        else:
            self.train_data_dir = os.path.join(self.cache_path, 'boxoban-levels-master', self.difficulty, self.split)

        if not os.path.exists(self.cache_path):
           
            url = "https://github.com/deepmind/boxoban-levels/archive/master.zip"
            
            if self.verbose:
                print('Boxoban: Pregenerated levels not downloaded.')
                print('Starting download from "{}"'.format(url))

            response = requests.get(url, stream=True)

            if response.status_code != 200:
                raise "Could not download levels from {}. If this problem occurs consistantly please report the bug under https://github.com/mpSchrader/gym-sokoban/issues. ".format(url)

            os.makedirs(self.cache_path)
            path_to_zip_file = os.path.join(self.cache_path, 'boxoban_levels-master.zip')
            with open(path_to_zip_file, 'wb') as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)

            zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
            zip_ref.extractall(self.cache_path)
            zip_ref.close()
        
        self.select_room(seed=seed)

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.get_image()

        return starting_observation, {}

    def select_map(self, seed=None):
        generated_files = [f for f in listdir(self.train_data_dir) if isfile(join(self.train_data_dir, f))]
        source_file = join(self.train_data_dir, random.choice(generated_files))

        maps = []
        current_map = []

        with open(source_file, 'r') as sf:
            for line in sf.readlines():
                if ';' in line and current_map:
                    maps.append(current_map)
                    current_map = []
                if '#' == line[0]:
                    current_map.append(line.strip())

        maps.append(current_map)

        if seed is not None:
            random.seed(seed)
        selected_map = random.choice(maps)

        if self.verbose:
            print('Selected Level from File "{}"'.format(source_file))
        return selected_map


    def select_room(self, seed=None):
        selected_map = self.select_map(seed=seed)
        self.room_fixed, self.room_state, self.box_mapping = self.generate_room(selected_map)


    def generate_room(self, select_map):
        room_fixed = []
        room_state = []

        targets = []
        boxes = []
        for row in select_map:
            room_f = []
            room_s = []

            for e in row:
                if e == '#':
                    room_f.append(0)
                    room_s.append(0)

                elif e == '@':
                    self.player_position = np.array([len(room_fixed), len(room_f)])
                    room_f.append(1)
                    room_s.append(5)


                elif e == '$':
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)

                elif e == '.':
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)

                else:
                    room_f.append(1)
                    room_s.append(1)

            room_fixed.append(room_f)
            room_state.append(room_s)


        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}

        return np.array(room_fixed), np.array(room_state), box_mapping





class FixedBoxobanEnv(BoxobanEnv):
    def select_room(self, seed=None) -> None:
        if not hasattr(self, "selected_map"):
            self.selected_map = self.select_map(seed=seed)
        self.room_fixed, self.room_state, self.box_mapping = self.generate_room(self.selected_map)
