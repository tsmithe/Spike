#include "RateAgent.hpp"

/*

  ScanWalkPolicy::choose_new_action
  ScanWalkTestPolicy::choose_new_action
  PlaceTestPolicy::choose_new_action
  HDTestPolicy::choose_new_action

 */

void RandomWalkPolicy::prepare(AgentBase& a) {
  FV_die = std::uniform_int_distribution<>(0, a.num_FV_states-1);
  AHV_die = std::uniform_int_distribution<>(0, a.num_AHV_states-1);
  prepared = true;
}

void RandomWalkPolicy::choose_new_action(AgentBase& a, FloatT dt) {
  // choose new action, ensuring legality
  // choice random with uniform distribution (for now)

  if (!prepared)
    prepare(a);
  
  bool is_legal = false;
  FloatT duration;
  while (!is_legal) {
    if (action_die(rand_engine) > p_fwd) {
      a.curr_action = AgentBase::actions_t::AHV;
      a.curr_FV = 0;

      a.curr_AHV = AHV_die(rand_engine);
      is_legal = true;

      FloatT angle_change = a.AHVs[a.curr_AHV].first * a.AHVs[a.curr_AHV].second;
      a.target_head_direction = a.head_direction + angle_change;
      duration = a.AHVs[a.curr_AHV].second;

      if (a.smooth_AHV) {
        FloatT total_AHV_change = a.AHVs[a.curr_AHV].first - a.AHV;
        FloatT AHV_change_time = total_AHV_change / a.AHV_speed;
        FloatT AHV_change_timesteps = AHV_change_time / dt;

        a.AHV_change = total_AHV_change / AHV_change_timesteps;
        a.change_action_ts = a.timesteps + AHV_change_timesteps;
        a.target_head_direction += a.AHV * AHV_change_time
          + 0.5 * a.AHV_speed * std::pow(AHV_change_time, 2.0f);
        duration += AHV_change_time;
      }
    } else {
      a.curr_action = AgentBase::actions_t::FV;
      a.curr_AHV = 0;
      a.curr_FV = FV_die(rand_engine);

      duration = a.FVs[a.curr_FV].second;
      FloatT FV = a.FVs[a.curr_FV].first;
      FloatT r = FV * duration;

      a.target_position = a.position;
      a.target_position(0) += r * cos(a.head_direction);
      a.target_position(1) += r * sin(a.head_direction);

      if ((fabs(a.target_position(0)) > a.bound_x)
          || (fabs(a.target_position(1)) > a.bound_y)) {
        // is_legal = false;

        a.curr_action = AgentBase::actions_t::AHV;
        a.curr_FV = 0;
        a.curr_AHV = AHV_die(rand_engine);

        is_legal = true;

        duration = M_PI / fabs(a.AHVs[a.curr_AHV].first);
        a.target_head_direction = a.head_direction + M_PI;

        if (a.target_head_direction > 2 * M_PI)
          a.target_head_direction -= 2 * M_PI;
      } else {
        is_legal = true;          
      }
    }
  }
  int timesteps_per_action = round(duration / dt);
  a.choose_next_action_ts = a.timesteps + timesteps_per_action;
}

void ScanWalkPolicy::prepare(AgentBase& a) {
  if (0 == bound_x)
    bound_x = a.bound_x;
  if (0 == bound_y)
    bound_y = a.bound_y;
  if (0 == row_separation)
    row_separation = bound_y / 5;
  walk_direction = 1;

  prepared = true;
}

void ScanWalkPolicy::choose_new_action(AgentBase& a, FloatT dt) {
  if (!prepared)
    prepare(a);

  if (a.choose_next_action_ts < 1) {
    a.position(0) = -bound_x;
    a.position(1) = bound_y;
  } else {
    a.position(1) -= 0.5 * row_separation * walk_direction;
  }

  if ((walk_direction > 0 && a.position(1) < -bound_y) ||
      (walk_direction < 0 && a.position(1) > bound_y)) {
    // a.position(1) = bound_y;
    a.position(1) += 0.5 * row_separation * walk_direction;
    walk_direction *= -1;
  }

  a.curr_action = AgentBase::actions_t::FV;
  a.curr_AHV = 0;
  a.curr_FV = 1;
  FloatT FV = a.FVs[a.curr_FV].first;

  if (a.position(0) >= bound_x) {
    a.head_direction = M_PI;
    a.target_position(0) = -bound_x;// * walk_direction;
  } else {
    a.head_direction = 0;
    a.target_position(0) = bound_x;// * walk_direction;
  }

  a.target_position(1) = a.position(1);

  FloatT duration = 2*bound_x / FV;
  int timesteps_per_action = round(duration / dt);
  a.choose_next_action_ts = a.timesteps + timesteps_per_action;
}

void ScanWalkPolicy::set_scan_bounds(FloatT x, FloatT y) {
  bound_x = x;
  bound_y = y;
}

void ScanWalkPolicy::set_row_separation(FloatT distance) {
  row_separation = distance;
}


void ScanWalkTestPolicy::choose_new_action(AgentBase& a, FloatT dt) {
  if (!prepared)
    prepare(a);

  if (!started) {
    old_position = a.position;
    old_target_pos = a.target_position;
    old_hd = a.head_direction;
    old_target_hd = a.target_head_direction;
    started = true;

    a.position(0) = -bound_x;
    a.position(1) = bound_y;
  } else {
    a.position(1) -= 0.5 * row_separation * walk_direction;
  }

  if ((walk_direction > 0 && a.position(1) < -bound_y) ||
      (walk_direction < 0 && a.position(1) > bound_y)) {
    // a.position(1) += 0.5 * row_separation * walk_direction;
    // walk_direction *= -1;
    started = false;

    a.test_times.pop();
    a.position = old_position;
    a.target_position = old_target_pos;
    a.target_head_direction = old_target_hd;

    a.curr_action = AgentBase::actions_t::STAY;
    a.curr_AHV = 0;
    a.curr_FV = 0;
    a.choose_next_action_ts = a.timesteps + 1;

    return;
  }

  a.curr_action = AgentBase::actions_t::FV;
  a.curr_AHV = 0;
  a.curr_FV = 1;
  FloatT FV = a.FVs[a.curr_FV].first;

  if (a.position(0) >= bound_x) {
    a.head_direction = M_PI;
    a.target_position(0) = -bound_x;// * walk_direction;
  } else {
    a.head_direction = 0;
    a.target_position(0) = bound_x;// * walk_direction;
  }

  a.target_position(1) = a.position(1);

  FloatT duration = 2*bound_x / FV;
  int timesteps_per_action = round(duration / dt);
  a.choose_next_action_ts = a.timesteps + timesteps_per_action;
}


void PlaceTestPolicy::choose_new_action(AgentBase& a, FloatT dt) {
  /* How does testing work?
     - For each test position (AHV):
       1. Select AHV
       2. Equilibrate (hold stationary for given time)
       3. Rotate through 4pi radians
     - For each test position, given an approach radius:
       - For each of a given number of equally spaced points
         on the approach radius:
         0. Equilibrate
         1. Set head direction
         2. Travel forward until reaching the opposite
            point on the circumference (bisecting the target)
            -- assert that first FV is positive for now
         3. Rotate by pi radians, and travel back.

     Once at last test stage, remove test_times.top(),
     and set curr_test_position to -1. */

  // NB: Currently, we only test one AHV and one FV
  //     -- assume that AHVs[0].first and FVs[0].first > 0
  assert(a.AHVs.size() > 0); assert(a.AHVs[1].first > 0);
  assert(a.FVs.size() > 0); assert(a.FVs[1].first > 0);

  const bool finished_place_test =
    (AgentBase::actions_t::FV == a.curr_action &&
     curr_test_approach_angle == test_approach_angles);

  if (AgentBase::actions_t::STAY == a.curr_action
      && curr_test_position >= 0) {
    if (curr_test_position < test_positions.size()) {
      // then we are on the AHV test
      // rotate through 4pi radians
      a.curr_action = AgentBase::actions_t::AHV;
      a.curr_AHV = 1;
      a.curr_FV = 0;
      a.target_head_direction = 0;
      a.target_position = a.position;
      FloatT duration = 4 * M_PI / fabs(a.AHVs[1].first);
      a.choose_next_action_ts = a.timesteps + round(duration / dt);
    } else if (curr_test_position < 2*test_positions.size()) {
      // then we are on the PLACE test, and have just finished
      // an equilibration
      //
      // so compute the bearing and the duration, and set
      // curr_action to FV
      a.curr_action = AgentBase::actions_t::FV;
      a.curr_FV = 1;
      a.curr_AHV = 0;
      FloatT radial_angle = 0.0 // M_PI / test_approach_angles
        + 2 * M_PI * curr_test_approach_angle / test_approach_angles;
      a.head_direction = radial_angle + M_PI;
      if (a.head_direction > 2 * M_PI) a.head_direction -= 2 * M_PI;
      EigenVector2D new_radial_position;
      new_radial_position(0) = test_approach_radius * cos(a.head_direction);
      new_radial_position(1) = test_approach_radius * sin(a.head_direction);
      a.target_position = test_positions[curr_test_position%test_positions.size()]
        + new_radial_position;
      a.target_head_direction = a.head_direction;
      FloatT duration = 2.0 * test_approach_radius / a.FVs[1].first;
      a.choose_next_action_ts = a.timesteps + round(duration / dt);

      // printf("\n@@@@@ STAY -> FV : %d, %d\n", curr_test_position, curr_test_approach_angle);
    } else {
      // printf("\n@@@@@ ??? : %d, %d\n", curr_test_position, curr_test_approach_angle);
      assert(false && "whoops -- logic error");
    }
  } else {
    if (AgentBase::actions_t::AHV == a.curr_action
        || finished_place_test
        || curr_test_position < 0) {
      // then we have just finished an action (rather than an
      // equilibration) and so should move to the next position.
      curr_test_position++;

      // update position
      //  -- nb: if at end of AHV, set position to the first
      //         position on the PLACE test radius
      if (curr_test_position >= test_positions.size()) {
        // then we are at the PLACE test, and so should set the
        // position (and heading) to first next position /on the
        // PLACE test radius/ !
        EigenVector2D radial_position;
        FloatT radial_angle = 0; // M_PI / test_approach_angles;
        radial_position(0) = test_approach_radius * cos(radial_angle);
        radial_position(1) = test_approach_radius * sin(radial_angle);
        a.position = test_positions[curr_test_position%test_positions.size()]
          + radial_position;
        a.head_direction = radial_angle + M_PI;
        if (a.head_direction > 2 * M_PI) a.head_direction -= 2 * M_PI;
        curr_test_approach_angle = 0;
      } else {
        a.position = test_positions[curr_test_position%test_positions.size()];
        a.head_direction = 0;
      }

      // set equilibration
      a.curr_action = AgentBase::actions_t::STAY;
      a.curr_FV = 0; a.curr_AHV = 0;
      a.target_position = a.position;
      a.target_head_direction = a.head_direction;
      a.choose_next_action_ts = a.timesteps + round(t_equilibration / dt);
    } else if (AgentBase::actions_t::FV == a.curr_action
               && !finished_place_test) {
      // then we are part-way through the PLACE test:
      // we have finished moving through the target location once
      // and are now either on the other side of the circumference
      //   or just finished coming back
      //
      // if on the other side:
      // + turn around (instantaneously...)
      // + go back, returning to next approach point
      // + set curr_test_approach_angle negative (as a marker)
      // if just back (ie, curr_test_approach_angle < 0):
      // + set curr_test_approach_angle +ve and increment
      // + set equilibration: we are at the next position

      // printf("\n@@@@@ FV -> STAY : %d, %d\n", curr_test_position, curr_test_approach_angle);
      /*
      if (curr_test_approach_angle >= 0) {
        // then on the other side
        FloatT this_radial_angle = 0.0 // M_PI / test_approach_angles
          + 2 * M_PI * curr_test_approach_angle / test_approach_angles;
        FloatT next_radial_angle = 0.0 // M_PI / test_approach_angles
          + 2 * M_PI * (curr_test_approach_angle + 1) / test_approach_angles;

        EigenVector2D this_radial_position;
        this_radial_position(0) = test_approach_radius * cos(this_radial_angle);
        this_radial_position(1) = test_approach_radius * sin(this_radial_angle);

        EigenVector2D next_radial_position;
        next_radial_position(0) = test_approach_radius * cos(next_radial_angle);
        next_radial_position(1) = test_approach_radius * sin(next_radial_angle);

        EigenVector2D chord = this_radial_position + next_radial_position;
        FloatT distance = chord.norm();
        head_direction = acos(chord(0) / distance);
        if (head_direction > 2 * M_PI) head_direction -= 2 * M_PI;

        target_position = next_radial_position;
        target_head_direction = head_direction;

        FloatT duration = distance / FVs[1].first;
        choose_next_action_ts = timesteps + round(duration / dt);

        curr_action = actions_t::FV;
        curr_FV = 1;
        curr_AHV = 0;

        curr_test_approach_angle = -(curr_test_approach_angle+1);
      } else {
        // then just back
        curr_test_approach_angle = abs(curr_test_approach_angle);
        FloatT radial_angle = 0.0 // M_PI / test_approach_angles
          + 2 * M_PI * curr_test_approach_angle / test_approach_angles;
        head_direction = radial_angle + M_PI;
        if (head_direction > 2 * M_PI) head_direction -= 2 * M_PI;
        target_position = position;
        target_head_direction = head_direction;

        // equilibrate:
        curr_action = actions_t::STAY;
        curr_FV = 0; curr_AHV = 0;
        choose_next_action_ts = timesteps + round(t_equilibration / dt);
      }
      */
      curr_test_approach_angle++;

      FloatT radial_angle = 0.0 // M_PI / test_approach_angles
        + 2 * M_PI * curr_test_approach_angle / test_approach_angles;
      a.position = test_positions[curr_test_position%test_positions.size()];
      a.position(0) += test_approach_radius * cos(radial_angle);
      a.position(1) += test_approach_radius * sin(radial_angle);
      a.head_direction = radial_angle + M_PI;
      if (a.head_direction > 2 * M_PI) a.head_direction -= M_PI;

      a.target_position = a.position;
      a.target_head_direction = a.head_direction;

      // equilibrate:
      a.curr_action = AgentBase::actions_t::STAY;
      a.curr_FV = 0; a.curr_AHV = 0;
      // TODO: Best equilibration time? Or use 'clever' algorithm?
      a.choose_next_action_ts = a.timesteps + round(0.2 * t_equilibration / dt);
    } else {
      // WHAT HAPPENS HERE? SHOULD NEVER REACH ...
      assert(false && "how did I get here?");
    }
  }

  if (2*test_positions.size() - 1 == curr_test_position
      && finished_place_test) {
    // printf("\n@@@@@ ??. : %d, %d\n", curr_test_position, curr_test_approach_angle);
    curr_test_position = -1;
    curr_test_approach_angle = 0;
    // TODO: set position and bearing to those at start of test period?
    a.test_times.pop();
  }
}


void HDTestPolicy::choose_new_action(AgentBase& a, FloatT dt) {
  /*
    At first: curr_test_position = -1
    So curr_test_position++ if < 0 or done with AHVs
    (And set equilibration)

    So we might have curr_test_position = end
    Let's ensure that doesn't happen

    So if we are do curr_test_position++ and the next time is invalid,
    pop the test time

    What if we are not done with AHVs?
    Then don't increment position index, but do increment AHV index

    In every case, set the target data and choose_next_action_ts
   */

  if (curr_test_position < 0) {
    old_position = a.position;
    old_target_pos = a.target_position;
    old_hd = a.head_direction;
    old_target_hd = a.target_head_direction;
  }

  if (curr_test_position < 0 || a.AHVs.size() <= a.curr_AHV+1) {
    curr_test_position++;
    a.curr_AHV = -1;
  }

  a.position = test_positions[curr_test_position];
  a.target_position = a.position;

  if (test_positions.size() <= curr_test_position+1) {
    curr_test_position = -1;
    a.test_times.pop();
    a.position = old_position;
    a.target_position = old_target_pos;
    // a.head_direction = old_hd;
    a.target_head_direction = old_target_hd;
  }

  a.curr_action = AgentBase::actions_t::AHV;
  a.curr_FV = 0;
  a.curr_AHV++;

  a.head_direction = 0;
  a.target_head_direction = 0;

  FloatT AHV = a.AHVs[a.curr_AHV].first;
  FloatT duration;
  if (a.curr_AHV == 0) {
    duration = 1.0;
  } else {
    duration = 2 * M_PI / fabs(AHV);
  }
  int timesteps_per_action = round(duration / dt);
  a.choose_next_action_ts = a.timesteps + timesteps_per_action;
}


/*
void Agent::connect_actor(RateNeurons* actor_) {
  actor = actor_;

  // NB: For now, the tuning of the actor is imposed,
  //     and assumes movement on a 2D plane.
  actor_tuning.resize(2, actor->size);
  FloatT angle = 0;
  for (int i = 0; i < actor->size; ++i) {
    Eigen::Matrix<FloatT, 2, 1> direction(2);
    direction(0) = cos(angle);
    direction(1) = sin(angle);
    actor_tuning.col(i) = direction;
    angle += 2 * M_PI / actor->size;
  }
}
*/

/*
void Agent::update_per_dt(FloatT dt) {
  /*
     + Turn actor into a velocity (assume a continuous state space)
       - this ~is the `output' of the RateModel
     + Set new position according to this velocity, dt, and the maze structure
     + Update the state vector according to the new position
       - NB this should probably be in AgentSenseRateNeurons
       - each element of the state vector measures `inverse distance' ??

     % NB: eventually, want to be able to represent self-embeddedness
   * /
  position.noalias() += (dt * velocity_scaling) * (actor_tuning * actor->rate());
  if (position(0) > bound_x)
    position(0) = bound_x;
  else if (position(0) < 0)
    position(0) = 0;

  if (position(1) > bound_y)
    position(1) = bound_y;
  else if (position(1) < 0)
    position(1) = 0;
}
*/



AgentVISRateNeurons::AgentVISRateNeurons(Context* ctx,
                                         AgentBase* agent_,
                                         int neurons_per_object_,
                                         FloatT sigma_IN_,
                                         FloatT lambda_,
                                         std::string label_)
  : RateNeurons(nullptr, (int)(agent_->num_objects) * neurons_per_object_,
                label_, 0, 1, 1),
    agent(agent_),
    neurons_per_object(neurons_per_object_),
    sigma_IN(sigma_IN_), lambda(lambda_) {

  theta_pref = EigenVector::Zero(neurons_per_object);
  for (int j = 0; j < neurons_per_object; ++j) {
    theta_pref(j) = 2 * M_PI * j/neurons_per_object;
  }

  if (ctx)
    init_backend(ctx);
}

AgentVISRateNeurons::~AgentVISRateNeurons() {
}


AgentHDRateNeurons::AgentHDRateNeurons(Context* ctx,
                                       AgentBase* agent_,
                                       int size_,
                                       FloatT sigma_IN_,
                                       FloatT lambda_,
                                       std::string label_)
  : RateNeurons(nullptr, size_, label_, 0, 1, 1),
    agent(agent_),
    sigma_IN(sigma_IN_), lambda(lambda_) {

  theta_pref = EigenVector::Zero(size);
  for (int j = 0; j < size; ++j)
    theta_pref(j) = 2 * M_PI * j/size;

  if (ctx)
    init_backend(ctx);
}

AgentHDRateNeurons::~AgentHDRateNeurons() {
}


AgentAHVRateNeurons::AgentAHVRateNeurons(Context* ctx,
                                         AgentBase* agent_,
                                         int neurons_per_state_,
                                         std::string label_)
  : RateNeurons(nullptr, 0, label_, 0, 1, 1), agent(agent_),
    neurons_per_state(neurons_per_state_) {

  // compute size:
  if (agent->smooth_AHV) {
    size = 3 * neurons_per_state_; // 1 symmetric class; 2 asymmetric
  } else {
    size = (int)(agent_->num_AHV_states) * neurons_per_state_;
  }

  RateNeurons(nullptr, size, label_, 0, 1, 1);

  if (ctx)
    init_backend(ctx);
}

AgentAHVRateNeurons::~AgentAHVRateNeurons() {
}

AgentFVRateNeurons::AgentFVRateNeurons(Context* ctx,
                                       AgentBase* agent_,
                                       int neurons_per_state_,
                                       std::string label_)
  : RateNeurons(nullptr, (int)(agent_->num_FV_states) * neurons_per_state_,
                label_, 0, 1, 1),
    agent(agent_),
    neurons_per_state(neurons_per_state_) {

  if (ctx)
    init_backend(ctx);
}

AgentFVRateNeurons::~AgentFVRateNeurons() {
}


SPIKE_MAKE_INIT_BACKEND(AgentVISRateNeurons);
SPIKE_MAKE_INIT_BACKEND(AgentHDRateNeurons);
SPIKE_MAKE_INIT_BACKEND(AgentAHVRateNeurons);
SPIKE_MAKE_INIT_BACKEND(AgentFVRateNeurons);
