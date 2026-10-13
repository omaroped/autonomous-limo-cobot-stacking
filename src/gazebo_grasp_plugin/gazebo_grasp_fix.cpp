// Holds a gripped object in Gazebo with a fixed joint between the gripper and the object.
// Based on GazeboGraspFix from Jennifer Buehler's gazebo_grasp_plugin
// (https://github.com/JenniferBuehler/gazebo-pkgs), GPLv3; license text in LICENSE.
// I use it so that the held cube is joined in the pose it has at contact and turns with the
// fingers. The weld used before in this project moved only the cube's position.
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>

#include <ignition/math.hh>

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gazebo {

class GazeboGraspFix : public ModelPlugin {
 public:
  GazeboGraspFix() : ModelPlugin() {}

  void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf) override {
    this->model_ = _parent;
    this->world_ = this->model_->GetWorld();
    this->physics_ = this->world_->Physics();

    if (_sdf->HasElement("gripper_finger_link")) {
      sdf::ElementPtr e = _sdf->GetElement("gripper_finger_link");
      while (e) {
        finger_links_.push_back(e->Get<std::string>());
        e = e->GetNextElement("gripper_finger_link");
      }
    }
    if (finger_links_.empty()) {
      gzwarn << "[GazeboGraspFix] No <gripper_finger_link> tags given — "
                "plugin will never detect a grasp.\n";
    }

    if (_sdf->HasElement("palm_link"))
      palm_link_name_ = _sdf->Get<std::string>("palm_link");
    else
      palm_link_name_ = "gripper_base";

    gripper_joint_name_ = _sdf->HasElement("gripper_joint")
                              ? _sdf->Get<std::string>("gripper_joint")
                              : "gripper_controller";
    release_position_ = _sdf->HasElement("release_position")
                            ? _sdf->Get<double>("release_position")
                            : 0.0;
    grasp_count_threshold_ = _sdf->HasElement("grasp_count_threshold")
                                 ? _sdf->Get<int>("grasp_count_threshold")
                                 : 10;

    node_ = transport::NodePtr(new transport::Node());
    node_->Init();

    contacts_sub_ = node_->Subscribe("~/physics/contacts",
                                     &GazeboGraspFix::OnContacts, this);

    update_conn_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&GazeboGraspFix::OnUpdate, this));

    gzmsg << "[GazeboGraspFix] loaded on model '" << this->model_->GetName()
          << "'  fingers=" << finger_links_.size()
          << "  palm='" << palm_link_name_
          << "'  gripper_joint='" << gripper_joint_name_
          << "'  release_pos=" << release_position_
          << "  threshold=" << grasp_count_threshold_ << "\n";
  }

 private:

  void OnContacts(const boost::shared_ptr<const msgs::Contacts> &_msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (int i = 0; i < _msg->contact_size(); ++i) {
      const msgs::Contact &c = _msg->contact(i);
      const std::string &c1 = c.collision1();
      const std::string &c2 = c.collision2();

      const bool f1 = IsFinger(c1);
      const bool f2 = IsFinger(c2);
      if (!f1 && !f2) continue;
      if (f1 && f2) continue;

      const std::string obj_collision = f1 ? c2 : c1;
      const std::string obj_model = ModelOf(obj_collision);
      if (obj_model.empty() || obj_model == model_->GetName()) continue;

      contact_accum_[obj_model] += 1;
    }
  }

  void OnUpdate() {
    std::lock_guard<std::mutex> lk(mtx_);

    auto this_tick = contact_accum_;
    contact_accum_.clear();

    std::unordered_set<std::string> seen;
    for (const auto &kv : this_tick) {
      const std::string &obj = kv.first;
      seen.insert(obj);
      if (kv.second > 0) {
        grip_count_[obj] += 1;
        release_count_[obj] = 0;
      }
    }

    for (auto &kv : grip_count_) {
      if (seen.find(kv.first) == seen.end()) {
        kv.second = 0;
        release_count_[kv.first] += 1;
      }
    }

    if (!held_obj_link_) {

      double gpos_attach = 0.0;
      physics::JointPtr gj_attach = model_->GetJoint(gripper_joint_name_);
      if (gj_attach) gpos_attach = gj_attach->Position(0);
      if (gpos_attach > release_position_) {

        grip_count_.clear();
        release_count_.clear();
        return;
      }

      std::string best_obj;
      int best_count = 0;
      for (const auto &kv : grip_count_) {
        if (kv.second >= grasp_count_threshold_ && kv.second > best_count) {

          physics::ModelPtr cand = world_->ModelByName(kv.first);
          if (!cand || cand->IsStatic()) continue;
          best_count = kv.second;
          best_obj = kv.first;
        }
      }
      if (!best_obj.empty()) AttachObject(best_obj);
    } else {

      double gpos = 0.0;
      physics::JointPtr gj = model_->GetJoint(gripper_joint_name_);
      if (gj) gpos = gj->Position(0);
      if (gpos > release_position_) {
        release_streak_ += 1;
        if (release_streak_ >= grasp_count_threshold_) ReleaseObject();
      } else {
        release_streak_ = 0;

        if (!held_obj_link_ || !held_obj_link_->GetParentModel())
          ReleaseObject();
      }
    }
  }

  void AttachObject(const std::string &obj_model_name) {
    physics::ModelPtr obj_model = world_->ModelByName(obj_model_name);
    if (!obj_model) {
      gzerr << "[GazeboGraspFix] object model '" << obj_model_name
            << "' not found in world\n";
      return;
    }

    auto links = obj_model->GetLinks();
    if (links.empty()) {
      gzerr << "[GazeboGraspFix] object '" << obj_model_name
            << "' has no links\n";
      return;
    }
    held_obj_link_ = links[0];

    physics::LinkPtr palm = model_->GetLink(palm_link_name_);
    if (!palm) {
      gzerr << "[GazeboGraspFix] palm link '" << palm_link_name_
            << "' not found on model\n";
      held_obj_link_.reset();
      return;
    }

    fixed_joint_ = physics_->CreateJoint("fixed", model_);
    fixed_joint_->Load(palm, held_obj_link_, ignition::math::Pose3d());
    fixed_joint_->Init();
    fixed_joint_->SetModel(model_);

    held_obj_link_->ResetPhysicsStates();
    held_obj_link_->SetLinearVel(ignition::math::Vector3d::Zero);
    held_obj_link_->SetAngularVel(ignition::math::Vector3d::Zero);

    held_obj_name_ = obj_model_name;

    grip_count_.clear();
    release_count_.clear();
    release_streak_ = 0;

    SetHeldCollision(true);
    gzmsg << "[GazeboGraspFix] GRASP ON — fixed joint '" << palm_link_name_
          << "' <-> '" << obj_model_name << "'\n";
  }

  void SetHeldCollision(bool held) {
    const unsigned int ALL = GZ_ALL_COLLIDE, BIT_OBJ = 0x01000000, BIT_HAND = 0x02000000;
    if (held_obj_link_) {
      for (auto &c : held_obj_link_->GetCollisions()) {
        c->SetCategoryBits(held ? BIT_OBJ : ALL);
        c->SetCollideBits(held ? (ALL & ~BIT_HAND) : ALL);
      }
    }
    for (auto &l : model_->GetLinks()) {
      const std::string n = l->GetName();
      if (n.find("gripper") == std::string::npos && n != palm_link_name_) continue;
      for (auto &c : l->GetCollisions()) {
        c->SetCategoryBits(held ? BIT_HAND : ALL);
        c->SetCollideBits(held ? (ALL & ~BIT_OBJ) : ALL);
      }
    }
  }

  void ReleaseObject() {
    SetHeldCollision(false);
    if (fixed_joint_) {
      fixed_joint_->Detach();
      fixed_joint_.reset();
    }
    if (held_obj_link_) {

      held_obj_link_.reset();
    }
    if (!held_obj_name_.empty()) {
      gzmsg << "[GazeboGraspFix] GRASP OFF — released '" << held_obj_name_
            << "'\n";
      held_obj_name_.clear();
    }
    grip_count_.clear();
    release_count_.clear();
  }

  bool IsFinger(const std::string &collision_scoped) const {
    for (const auto &fl : finger_links_)
      if (collision_scoped.find(fl) != std::string::npos) return true;
    return false;
  }

  static std::string ModelOf(const std::string &scoped) {
    const auto p = scoped.find("::");
    if (p == std::string::npos) return "";
    return scoped.substr(0, p);
  }

  physics::ModelPtr model_;
  physics::WorldPtr world_;
  physics::PhysicsEnginePtr physics_;

  std::vector<std::string> finger_links_;
  std::string palm_link_name_;
  std::string gripper_joint_name_;
  double release_position_;
  int grasp_count_threshold_;
  int release_streak_ = 0;

  transport::NodePtr node_;
  transport::SubscriberPtr contacts_sub_;
  event::ConnectionPtr update_conn_;

  std::mutex mtx_;

  std::unordered_map<std::string, int> contact_accum_;
  std::unordered_map<std::string, int> grip_count_;
  std::unordered_map<std::string, int> release_count_;

  physics::JointPtr fixed_joint_;
  physics::LinkPtr held_obj_link_;
  std::string held_obj_name_;
};

GZ_REGISTER_MODEL_PLUGIN(GazeboGraspFix)

}
