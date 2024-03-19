import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces import Box
from .room_utils import generate_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np
from pathlib import Path

class SokobanEnv(gym.Env):
    metadata = {
        'render_modes': ['rgb_array'],
        'render_fps': 4
    }
    kWall = 0
    kEmpty = 1
    kTarget = 2
    kBoxOnTarget = 3
    kBox = 4
    kPlayer = 5
    kPlayerOnTarget = 6
    kPrintLevelKey = "# .s$@a"

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 render_mode='rgb_array',
                 tinyworld_obs=False,
                 tinyworld_render=False,
                 reset=True,
                 terminate_on_first_box=False,
                 reset_seed = None,
                 levels_dir=None,
                 load_sequentially=False,
                 n_levels_to_load=-1,
                 verbose=0
                ):
        self.terminate_on_first_box = terminate_on_first_box

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Rendering variables
        self.render_mode = render_mode
        self.tinyworld_render = tinyworld_render

        self.window = None
        self.clock = None

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

        # Other Settings
        assert render_mode in self.metadata["render_modes"], f"Unknown Rendering Mode {render_mode}"
        self.use_tiny_world = tinyworld_obs
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        sprite_sz = 1 if self.use_tiny_world else 16
        screen_height, screen_width = (dim_room[0] * sprite_sz, dim_room[1] * sprite_sz)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)

        self.seed(reset_seed)
        self.load_sequentially = load_sequentially
        self.verbose = verbose
        self.n_levels_to_load = n_levels_to_load
        if self.n_levels_to_load > 0 or self.load_sequentially:
            assert levels_dir is not None, "Levels directory must be specified if n_levels_to_load > 0 or load_sequentially"
        self.levels_dir = levels_dir
        if self.levels_dir is not None:
            self.levels_dir = Path(self.levels_dir)
            assert self.levels_dir.is_dir(), f"Levels directory {self.levels_dir} is not a directory"
            self.level_files = np.array(list(self.levels_dir.glob("*.txt")))
            assert len(self.level_files) > 0, f"Levels directory {self.levels_dir} does not contain any .txt files"
            self.level_files.sort()
            if not self.load_sequentially:
                self.level_file_shuffle_order = self.np_random.permutation(len(self.level_files))
                self.level_files = self.level_files[self.level_file_shuffle_order]
            self.level_file_idx = -1
            self.level_idx = -1
            self.levels = []
        if reset:
            # Initialize Room
            _ = self.reset(reset_seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert isinstance(action, int) or action.shape == ()

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 4]
        elif action < 5:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.get_image()

        info = {
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()
        return observation, self.reward_last, done, False, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.
        return ((self.terminate_on_first_box and self.boxes_on_target > 0)
                or self._check_if_all_boxes_on_target()
                or self._check_if_maxsteps())

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)

    def print_level(self, level: np.ndarray):
        dim_room = level.shape[0]
        if dim_room == 0:
            raise RuntimeError("dim_room cannot be zero.")
        for r in range(len(level)):
            for c in range(len(level[r])):
                print(self.kPrintLevelKey[level[r, c]], end='')
            print()

    def add_line(self, level: np.ndarray, row_num: int, line: str):
        start = line[0]
        end = line[-1]
        if start != '#' or end != '#':
            raise RuntimeError(f"Line '{line}' does not start ({start}) and end ({end}) with '#', as it should.")

        for col_num, r in enumerate(line):
            if r == '#':
                level[row_num, col_num] = self.kWall
            elif r == '@':
                level[row_num, col_num] = self.kPlayer
            elif r == '$':
                level[row_num, col_num] = self.kBox
            elif r == '.':
                level[row_num, col_num] = self.kTarget
            elif r == ' ':
                level[row_num, col_num] = self.kEmpty
            else:
                raise RuntimeError(f"Line '{line}' has character '{r}' which is not in the valid set '#@$. '.")

    def load_level_file(self, file_path):
        print("loading new file:", file_path)
        assert self.level_idx >= len(self.levels), f"Level index {self.level_idx} is not at the end of levels list {len(self.levels)}"
        self.levels = []
        with open(file_path, 'r') as file:
            cur_level = []
            for line in file:
                line = line.strip()
                if line == '':
                    continue
                if line[0] == '#':
                    # Count contiguous '#' characters and use this as the box dimension
                    dim_room = line.count('#')
                    assert dim_room == self.dim_room[0] == self.dim_room[1], f"Level dimension {dim_room} does not match dim_room={self.dim_room}"
                    cur_level = np.zeros((dim_room, dim_room), dtype=np.int8)
                    row_num = 0
                    self.add_line(cur_level, row_num, line)

                    for line in file:
                        line = line.strip()
                        if line == '' or line[0] != '#':
                            break
                        if len(line) != dim_room:
                            raise RuntimeError(f"Irregular line '{line}' does not match dim_room={dim_room}")
                        row_num += 1
                        self.add_line(cur_level, row_num, line)

                    if cur_level.shape[0] != dim_room or cur_level.shape[1] != dim_room:
                        raise RuntimeError(f"Room is not square: {len(cur_level)} != {dim_room}x{dim_room}")
                    self.levels.append(cur_level)

            self.levels = np.array(self.levels)
            if not self.load_sequentially:
                self.levels_shuffle_order = self.np_random.permutation(len(self.levels))
                self.levels = self.levels[self.levels_shuffle_order]
            if len(self.levels) == 0:
                raise RuntimeError(f"No levels loaded from file '{file_path}'")

            if self.verbose >= 1:
                print(f"***Loaded {len(self.levels)} levels from {file_path}")
                if self.verbose >= 2:
                    self.print_level(self.levels[0])
                    print()
                    self.print_level(self.levels[1])
                    print()


    def reset(self, seed=None, options={}, second_player=False, render_mode='rgb_array'):
        if self.levels_dir is not None:
            self.level_idx += 1
            if self.level_idx >= len(self.levels):
                self.level_file_idx += 1
                if self.level_file_idx >= len(self.level_files):
                    if not self.load_sequentially:
                        # after every full pass through the levels, reshuffle them
                        self.level_files.sort()
                        self.level_file_shuffle_order = self.np_random.permutation(len(self.level_files))
                        self.level_files = self.level_files[self.level_file_shuffle_order]
                    self.level_file_idx = 0
                print(self.level_idx, self.level_file_idx)
                level_file = self.level_files[self.level_file_idx]
                self.load_level_file(level_file)
                self.level_idx = 0
                
            self.room_state = self.levels[self.level_idx]
            self.room_fixed = self.room_state.copy()
            self.room_fixed[self.room_fixed == self.kBox] = self.kEmpty
            self.room_fixed[self.room_fixed == self.kPlayer] = self.kEmpty
            self.box_mapping = None
        else:
            try:
                self.room_fixed, self.room_state, self.box_mapping = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    second_player=second_player
                )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                return self.reset(seed, second_player=second_player, render_mode=render_mode)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.get_image()
        return starting_observation, {}

    def render(self):
        img = self.get_image(use_tiny_world=self.tinyworld_render)
        return img

    def get_image(self, use_tiny_world: bool | None = None):
        use_tiny_world = (self.use_tiny_world if use_tiny_world is None else use_tiny_world)
        if use_tiny_world:
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)
        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']
