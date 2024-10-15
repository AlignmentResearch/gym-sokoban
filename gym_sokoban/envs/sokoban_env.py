import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces import Box
from .room_utils import generate_room, generate_custom_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np
import random

class SokobanEnv(gym.Env):
    metadata = {
        'render_modes': ['rgb_array', "rgb_8x8"],
        'render_fps': 4
    }

    def __init__(self,
        dim_room=(10, 10),
        max_steps=120,
        min_episode_steps=60,
        num_boxes=4,
        num_gen_steps=None,
        render_mode='rgb_array',
        tinyworld_obs=False,
        tinyworld_render=False,
        tinyworld_scale=1,
        reset=True,
        terminate_on_first_box=False,
        reset_seed = None,
        reward_finished = 10,
        reward_box_on_target = 1,
        penalty_box_off_target = -1,
        penalty_for_step = -0.1,
    ):
        self.min_episode_steps = min_episode_steps
        if max_steps < self.min_episode_steps:
            raise ValueError(f"{max_steps=} cannot be less than {min_episode_steps=}")
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
        self.tinyworld_scale = tinyworld_scale

        self.window = None
        self.clock = None

        # Penalties and Rewards
        self.reward_finished = reward_finished
        self.reward_box_on_target = reward_box_on_target
        self.penalty_box_off_target = penalty_box_off_target
        self.penalty_for_step = penalty_for_step
        self.reward_last = 0

        # Other Settings
        assert render_mode in self.metadata["render_modes"], f"Unknown Rendering Mode {render_mode}"
        self.use_tiny_world = tinyworld_obs
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        if self.use_tiny_world:
            sprite_sz = self.tinyworld_scale
        elif self.render_mode == 'rgb_8x8':
            sprite_sz = 8
        else:
            sprite_sz = 16
        screen_height, screen_width = (dim_room[0] * sprite_sz, dim_room[1] * sprite_sz)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        self.this_episode_steps = max_steps

        self.seed(reset_seed)
        if reset:
            # Initialize Room
            _ = self.reset(reset_seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def next_observations(self):
        actions = []
        observations = []
        for action in ACTION_LOOKUP:
            can_move, _, (_, room_fixed, room_state) = self.take_action(action)
            if self.use_tiny_world:
                obs = room_to_tiny_world_rgb(room_state, room_fixed, self.tinyworld_scale)
            elif self.render_mode.startswith('rgb'):
                obs = room_to_rgb(room_state, room_fixed, is_8x8=self.render_mode == 'rgb_8x8')
            else:
                raise ValueError(f"Unknown Rendering Mode {self.render_mode}")
            if can_move:
                observations.append(obs)
                actions.append(action)
        return actions, observations

    def step(self, action):
        assert isinstance(action, int) or action.shape == ()

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        moved_player, moved_box = self._push(action)

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
            info["is_success"] = done

        return observation, self.reward_last, done, (not done) and self._check_if_maxsteps(), info

    def take_action(self, action):
        change = CHANGE_COORDINATES[action]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1] \
                or new_box_position[0] < 0 or new_box_position[1] < 0:
            return False, False, (current_position, self.room_fixed, self.room_state)


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        can_move = self.room_state[new_position[0], new_position[1]] in [1, 2]
        can_move |= can_push_box

        new_room_state = self.room_state.copy()
        new_room_fixed = self.room_fixed.copy()
        updated_player_position = current_position
        if can_move:
            updated_player_position = new_position
            new_room_state[(new_position[0], new_position[1])] = 5
            new_room_state[current_position[0], current_position[1]] = \
                new_room_fixed[current_position[0], current_position[1]]

        if can_push_box:
            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Box
            box_type = 4
            if new_room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            new_room_state[new_box_position[0], new_box_position[1]] = box_type
        return can_move, can_push_box, (updated_player_position, new_room_fixed, new_room_state)

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        can_move, can_push_box, new_pos_and_room = self.take_action(action)
        self.player_position, self.room_fixed, self.room_state = new_pos_and_room

        return can_move, can_push_box


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
        # Check if the game is over by pushing all boxes on the targets.
        return self._check_if_all_boxes_on_target() or \
                ((self.terminate_on_first_box and self.boxes_on_target > 0))

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.this_episode_steps == self.num_env_steps)

    def reset(self, seed=None, options={}, second_player=False, render_mode='rgb_array'):
        custom_level = False
        if "walls" in options:
            for k in ["walls", "boxes", "targets", "player"]:
                assert k in options
            custom_level = True
            self.set_custom_map(options["walls"], options["boxes"], options["targets"], options["player"])
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

        self.this_episode_steps = self.np_random.integers(self.min_episode_steps, self.max_steps+1).item()

        try:
            self.player_position = np.argwhere(self.room_state == 5)[0]
        except IndexError:
            assert custom_level, "player position can only be different from 5 in custom levels"
            self.player_position = np.argwhere(self.room_state == 6)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.get_image()
        return starting_observation, {}
    
    def set_custom_map(self, walls, boxes, targets, player):
        self.room_fixed, self.room_state = generate_custom_room(walls, boxes, targets, player, dim=self.dim_room)
        self.box_mapping = None
        return

    def render(self):
        img = self.get_image(use_tiny_world=self.tinyworld_render)
        return img

    def get_image(self, use_tiny_world: bool | None = None):
        use_tiny_world = (self.use_tiny_world if use_tiny_world is None else use_tiny_world)
        if use_tiny_world:
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, self.tinyworld_scale)
        elif self.render_mode.startswith('rgb'):
            img = room_to_rgb(self.room_state, self.room_fixed, is_8x8=self.render_mode == 'rgb_8x8')
        else:
            raise ValueError(f"Unknown Rendering Mode {self.render_mode}")
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
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
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
