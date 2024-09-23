import torch  # Must import before Open3D when using CUDA!
from torch.utils.data import DataLoader

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

import time
import threading

from como.data.odom_datasets import odom_collate_fn
from como.utils.o3d import (
    torch_to_o3d_rgb,
    torch_to_o3d_depth,
    pose_to_camera_setup,
    torch_to_o3d_rgb_with_points,
    torch_to_o3d_spheres,
    frustum_lineset,
    get_one_way_lineset,
)
from como.utils.io import save_traj
from como.utils.config import str_to_dtype

import OpenGL.GL as gl
import glfw
import cv2
from como.gui.OpenGLRenderer import OpenGLRenderer


class GuiWindow:
    def __init__(self, viz_cfg, slam_cfg, dataset):
        self.is_live = dataset.is_live  # Affects how data is loaded
        self.cfg = viz_cfg
        self.device = self.cfg["device"]
        self.dtype = str_to_dtype(self.cfg["dtype"])

        self.window_w, self.window_h = 2490, 1536 # 1920, 1080
        self.window = gui.Application.instance.create_window(
            "COMO", width=self.window_w, height=self.window_h
        )
        em = 10

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(
            left=spacing, top=vspacing, right=spacing, bottom=vspacing
        )

        self.ctrl_panel = gui.Vert(spacing, margins)

        ## Application control

        # Resume/pause
        resume_button = gui.ToggleSwitch("Resume/Pause")
        resume_button.set_on_clicked(self._on_pause_switch)
        resume_button.is_on = True

        # View control
        follow_button = gui.ToggleSwitch("Follow Tracking")
        follow_button.set_on_clicked(self._on_follow_switch)
        follow_button.is_on = True
        self.follow_tracking = follow_button.is_on

        # Next frame
        self.idx_panel = gui.Horiz(em)
        self.idx_label = gui.Label("Idx: {:20d}".format(0))
        next_frame_button = gui.Button("Next frame")
        next_frame_button.vertical_padding_em = 0.0
        next_frame_button.set_on_clicked(self._on_press)
        self.idx_panel.add_child(self.idx_label)
        self.idx_panel.add_child(next_frame_button)

        base_pose_button = gui.Button("Reset base pose")
        base_pose_button.vertical_padding_em = 0.0
        base_pose_button.set_on_clicked(self._on_press2)

        # save_traj_button = gui.Button("Save traj")
        # save_traj_button.vertical_padding_em = 0.0
        # save_traj_button.set_on_clicked(self._on_press3)

        # Geometry visualization
        self.anchor_point_slider = gui.Slider(gui.Slider.DOUBLE)
        self.anchor_point_slider.set_limits(-4.0, -1.0)  # Log of radius
        self.anchor_point_slider.double_value = -2.5

        self.render_lv = gui.ListView()
        render_options = ["None", "Point Cloud", "Phong"]
        self.render_lv.set_items(render_options)
        self.render_lv.selected_index = (
            self.render_lv.selected_index + 3
        )  # initially is -1, so now 1
        self.render_lv.set_max_visible_items(2)
        self.render_lv.set_on_selection_changed(self._on_list)
        self.render_val = render_options[self.render_lv.selected_index]

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)

        ### Data tab
        tab_data = gui.Vert(0, tab_margins)
        self.curr_rgb_w = gui.ImageWidget()
        self.kf_rgb_w = gui.ImageWidget()
        self.kf_depth_w = gui.ImageWidget()
        tab_data.add_child(self.curr_rgb_w)
        tab_data.add_fixed(vspacing)
        tab_data.add_child(self.kf_rgb_w)
        tab_data.add_fixed(vspacing)
        tab_data.add_child(self.kf_depth_w)

        ### Add panel children
        self.ctrl_panel.add_child(resume_button)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(self.idx_panel)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(follow_button)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(base_pose_button)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(self.render_lv)
        # self.ctrl_panel.add_fixed(vspacing)
        # self.ctrl_panel.add_child(save_traj_button)
        self.ctrl_panel.add_fixed(vspacing)
        self.ctrl_panel.add_child(gui.Label("Log10 Anchor Radius"))
        self.ctrl_panel.add_child(self.anchor_point_slider)
        self.ctrl_panel.add_fixed(vspacing)

        self.ctrl_panel.add_child(tab_data)

        self.widget3d = gui.SceneWidget()

        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label("FPS: 0.0")
        self.fps_panel.add_child(self.output_fps)

        self.num_tracked_panel = gui.Vert(spacing, margins)
        self.num_tracked_label = gui.Label("# Tracked Points:  0")
        self.num_tracked_panel.add_child(self.num_tracked_label)

        self.window.add_child(self.ctrl_panel)
        self.window.add_child(self.widget3d)
        self.window.add_child(self.fps_panel)
        self.window.add_child(self.num_tracked_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 0])
        self.widget3d.scene.scene.enable_sun_light(True)

        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        # Application variables
        self.is_running = resume_button.is_on
        self.is_done = False
        self.advance_one_frame = False

        # Point cloud mat
        self.pcd_mat = rendering.MaterialRecord()
        self.pcd_mat.point_size = self.cfg["pcd_point_size"]
        self.pcd_mat.shader = self.cfg["pcd_shader"]

        # Sparse point cloud mat
        self.sparse_pcd_mat = rendering.MaterialRecord()
        self.sparse_pcd_mat.point_size = 10.0
        self.sparse_pcd_mat.shader = self.cfg["pcd_shader"]

        # Line mat
        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 2.0
        self.line_mat.transmission = 1.0

        self.scale = 1.0
        self.base_pose = torch.eye(4)

        # Start processes
        torch.multiprocessing.set_start_method("spawn")
        self.dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            collate_fn=odom_collate_fn,
        )
        self.setup_slam_processes(slam_cfg)
        self.init_opengl()

        # Locks for thread safety
        self.kf_window_lock = threading.Lock()
        
        self.update_curr_image_lock = threading.Lock()
        self.update_curr_image_done = True
        self.update_pose_render_lock = threading.Lock()
        self.update_pose_render_done = True
        self.render_o3d_lock = threading.Lock()
        self.render_o3d_done = True
        self.update_keyframe_lock = threading.Lock()
        self.update_keyframe_done = True

        # Start running
        threading.Thread(name="UpdateMain", target=self.update_main).start()

    def init_glfw(self):
        window_name = "headless rendering"
        if not glfw.init():
            print("glfw not init")
            exit(1)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(
            self.window_w, self.window_h, window_name, None, None
        )
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        if not window:
            print("glfw window failed")
            glfw.terminate()
            exit(1)
        return window

    def init_opengl(self):
        self.window_gl = self.init_glfw()
        img_size = self.get_img_size()
        self.g_renderer = OpenGLRenderer(img_size[1], img_size[0])

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_BLEND)

        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

    def get_current_cam(self):
        P = self.widget3d.scene.camera.get_projection_matrix()
        V = self.widget3d.scene.camera.get_view_matrix()
        # proj_model_view_mat = P @ V
        # T = torch.from_numpy(proj_model_view_mat)
        return torch.from_numpy(V), torch.from_numpy(P)  #  T = P @ V

    def update_activated_renderer_state(self, kf_rgbs, kf_depths, kf_poses, V, P):
        B = kf_rgbs.shape[0]
        K = self.get_intrinsics()
        for b in range(B):
            self.g_renderer.render_keyframe(
                kf_rgbs[b], kf_depths[b], kf_poses[b], K, V, P
            )

    def render_o3d_image(self):
        # t1 = time.time()

        V, P = self.get_current_cam()

        glfw.poll_events()
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        w = self.widget3d_width
        h = self.widget3d_height
        glfw.set_window_size(self.window_gl, w, h)
        gl.glViewport(0, 0, w, h)

        # Since this is queued to be called by main thread, need to lock to avoid updates in the middle of function call
        self.kf_window_lock.acquire()
        self.update_activated_renderer_state(
            self.kf_rgb_window, self.kf_depth_window, self.kf_poses_window, V, P
        )
        self.kf_window_lock.release()

        width, height = glfw.get_framebuffer_size(self.window_gl)
        bufferdata = gl.glReadPixels(
            0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
        )
        img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
        cv2.flip(img, 0, img)
        render_img = o3d.geometry.Image(img)
        glfw.swap_buffers(self.window_gl)

        # Set background to render
        self.widget3d.scene.set_background([0, 0, 0, 1], render_img)

        # self.render_o3d_lock.acquire()
        self.render_o3d_done = True
        # self.render_o3d_lock.release()

        # t2 = time.time()
        # print("render_o3d: ", t2-t1)

        return

    def _on_press(self):
        self.advance_one_frame = True

    def _on_press2(self):
        self.base_pose = self.last_pose

    def _on_press3(self):
        self.save_traj()

    def _on_list(self, new_val, is_dbl_click):
        self.render_val = new_val

        if self.render_val != "Phong":
            gui.Application.instance.post_to_main_thread(
                self.window, lambda: self.widget3d.scene.set_background([1, 1, 1, 0])
            )

    def _on_layout(self, ctx):
        em = ctx.theme.font_size
        panel_width = 20 * em
        rect = self.window.content_rect

        self.ctrl_panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.ctrl_panel.frame.get_right()
        self.widget3d_width = rect.get_right() - x
        self.widget3d_height = rect.height
        self.widget3d.frame = gui.Rect(
            x, rect.y, self.widget3d_width, self.widget3d_height
        )

        fps_panel_width = 7 * em
        fps_panel_height = 2 * em
        self.fps_panel.frame = gui.Rect(
            rect.get_right() - fps_panel_width,
            rect.y,
            fps_panel_width,
            fps_panel_height,
        )

        inducing_panel_width = 10 * em
        inducing_panel_height = 2 * em
        self.num_tracked_panel.frame = gui.Rect(
            self.ctrl_panel.frame.get_right(),
            rect.y,
            inducing_panel_width,
            inducing_panel_height,
        )

    # Toggle callback: application's main controller
    def _on_pause_switch(self, is_on):
        self.is_running = is_on

    def _on_follow_switch(self, is_on):
        self.follow_tracking = is_on

    def update_kf_vars(self, kf_timestamps, kf_rgb, kf_depth, kf_poses, P):
        # t1 = time.time()

        # Local window for visualization - Need to lock since used for shading
        self.kf_window_lock.acquire()
        self.kf_rgb_window = kf_rgb
        self.kf_depth_window = kf_depth
        self.kf_poses_window = kf_poses
        self.P = P
        self.kf_window_lock.release()

        # t2 = time.time()
        # print("update_kf_vars: ", t2-t1)

        # # Full history
        # kf_timestamps_tensor = torch.as_tensor(kf_timestamps, dtype=torch.double)

        # # Handle first one
        # if self.kf_timestamps[0] < 0.0:
        #     self.kf_window_start_ind = 0
        # elif kf_timestamps_tensor[0] != self.kf_timestamps[self.kf_window_start_ind]:
        #     self.kf_window_start_ind += 1

        # i = self.kf_window_start_ind
        # self.kf_window_end_ind = i + kf_timestamps_tensor.shape[0]
        # j = self.kf_window_end_ind

        # self.kf_timestamps[i:j] = kf_timestamps_tensor
        # self.kf_rgb[i:j] = kf_rgb
        # self.kf_depth[i:j] = kf_depth
        # self.kf_poses[i:j] = kf_poses

    # def save_traj(self):
    #     if self.dataloader.dataset.seq_path is not None:
    #         filename = "./results/" + self.dataloader.dataset.save_traj_name + ".txt"
    #         j = self.kf_window_end_ind
    #         kf_timestamps = self.kf_timestamps[:j].tolist()
    #         kf_poses = self.kf_poses[:j]
    #         save_traj(filename, kf_timestamps, kf_poses)

    #         print("Saved trajectory.")

    def _on_start(self):
        pass

    def _on_close(self):
        self.is_done = True

        # self.save_traj()

        return True

    def setup_camera_view(self, pose, base_pose):
        center, eye, up = pose_to_camera_setup(pose, base_pose, self.scale)
        self.widget3d.look_at(center, eye, up)

    def get_intrinsics(self):
        return self.dataloader.dataset.intrinsics

    def get_img_size(self):
        img_size = self.dataloader.dataset.img_size
        return img_size

    def init_render(self, rgb):
        rgb_o3d = torch_to_o3d_rgb(rgb[0, ...])
        self.curr_rgb_w.update_image(rgb_o3d.to_legacy())

        rgb_o3d = torch_to_o3d_rgb(torch.ones_like(rgb[0, ...]))
        self.kf_rgb_w.update_image(rgb_o3d.to_legacy())

        kf_depth_img = torch_to_o3d_depth(torch.ones_like(rgb[0, 0:1, ...]))
        kf_depth_color = kf_depth_img.colorize_depth(
            self.cfg["depth_scale"], self.cfg["depth_min"], self.cfg["depth_max"]
        )
        self.kf_depth_w.update_image(kf_depth_color.to_legacy())

        self.window.set_needs_layout()

        # Way to set FoV of rendering
        fov = 40.0
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(fov, bounds, bounds.get_center())

        self.frumstum_scale = self.cfg["frustum_const"]

    def update_idx_text(self):
        self.idx_label.text = "Idx: {:8d}".format(self.idx)

    def update_curr_image_render(self, rgb):
        rgb_o3d = torch_to_o3d_rgb(rgb[0, ...])
        self.curr_rgb_w.update_image(rgb_o3d.to_legacy())

        # self.update_curr_image_lock.acquire()
        self.update_curr_image_done = True
        # self.update_curr_image_lock.release()

        return

    def update_keyframe_render(
        self,
        kf_timestamps,
        kf_rgbs,
        kf_poses,
        kf_depths,
        kf_sparse_coords,
        P_sparse,
        obs_ref_mask,
        one_way_poses,
        kf_pairs,
        one_way_pairs,
        pcd,
        kf_normals,
    ):  # Optional arguments

        # t1 = time.time()

        num_inducing_pts = kf_sparse_coords.shape[1]
        # colors = torch.tensor(self.cfg["inducing_point_color"], device=self.device).repeat(num_inducing_pts,1)
        colors_old = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        color_new = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        obs_ref_mask = obs_ref_mask.bool()
        colors = colors_old[None, :].repeat(num_inducing_pts, 1)
        colors[obs_ref_mask[-1, :], :] = color_new
        # colors = kf_sparse_coords.shape[1] * [self.cfg["inducing_point_color"]]
        sparse_coords = kf_sparse_coords[-1, None, :, :]
        coord_mask = (sparse_coords > 0).all(dim=2).flatten()
        sparse_coords = sparse_coords[:, coord_mask, :]
        kf_rgb_o3d = torch_to_o3d_rgb_with_points(
            kf_rgbs[-1, ...],
            sparse_coords,
            radius=self.cfg["inducing_point_radius"],
            colors=colors,
        )

        num_tracked = num_inducing_pts - torch.sum(obs_ref_mask[-1, :])
        self.num_tracked_label.text = "# Tracked Points: " + str(num_tracked.item())

        self.kf_rgb_w.update_image(kf_rgb_o3d.to_legacy())

        last_kf_depth = kf_depths[-1, ...]
        self.scale = torch.median(last_kf_depth).item()
        kf_depth_img = torch_to_o3d_depth(last_kf_depth.float())

        depth_min = self.cfg["depth_min"]
        depth_max = self.cfg["depth_max"]

        kf_depth_color = kf_depth_img.colorize_depth(self.scale, depth_min, depth_max)
        self.kf_depth_w.update_image(kf_depth_color.to_legacy())

        # Show sparse point cloud
        sparse_point_color = np.array([[1.0], [0.0], [0.0]])
        anchor_radius = self.scale*np.power(10.0, self.anchor_point_slider.double_value)
        spheres = torch_to_o3d_spheres(
            P_sparse, anchor_radius, resolution=5, color=sparse_point_color
        )
        self.widget3d.scene.remove_geometry("sparse_points")
        self.widget3d.scene.add_geometry("sparse_points", spheres, self.sparse_pcd_mat)

        self.frumstum_scale = self.cfg["frustum_const"] * self.scale
        # Keyframe window
        kf_geo = get_one_way_lineset(
            kf_poses.numpy(),
            kf_poses.numpy(),
            kf_pairs,
            self.get_intrinsics(),
            self.get_img_size(),
            self.cfg["kf_color"],
            frustum_scale=self.frumstum_scale,
        )
        self.widget3d.scene.remove_geometry("kf_frames")
        self.widget3d.scene.add_geometry("kf_frames", kf_geo, self.line_mat)
        # One-way frames
        if one_way_poses.shape[0] > 0:
            one_way_geo = get_one_way_lineset(
                kf_poses.numpy(),
                one_way_poses.numpy(),
                one_way_pairs,
                self.get_intrinsics(),
                self.get_img_size(),
                self.cfg["one_way_color"],
                frustum_scale=self.frumstum_scale,
            )
            self.widget3d.scene.remove_geometry("one_way_frames")
            self.widget3d.scene.add_geometry(
                "one_way_frames", one_way_geo, self.line_mat
            )

        if pcd is not None:
            self.widget3d.scene.remove_geometry("est_points")
            self.widget3d.scene.add_geometry("est_points", pcd, self.pcd_mat)
        else:
            self.widget3d.scene.remove_geometry("est_points")

        # self.update_keyframe_lock.acquire()
        self.update_keyframe_done = True
        # self.update_keyframe_lock.release()

        # t2 = time.time()
        # print("update_kf_render: ", t2-t1)

        return

    def update_pose_render(self, tracked_pose):
        pose_np = tracked_pose[0, :, :].numpy()
        est_traj_geo = frustum_lineset(
            self.get_intrinsics(),
            self.get_img_size(),
            pose_np,
            scale=self.frumstum_scale,
        )

        # t1 = time.time()

        est_traj_geo.paint_uniform_color(self.cfg["tracking_color"])
        self.widget3d.scene.remove_geometry("est_traj")
        self.widget3d.scene.add_geometry("est_traj", est_traj_geo, self.line_mat)

        if self.follow_tracking:
            self.setup_camera_view(pose_np, self.base_pose)
        else:
            pass

        # self.update_pose_render_lock.acquire()
        self.update_pose_render_done = True
        # self.update_pose_render_lock.release()

        # t2 = time.time()
        # print("update_pose_render: ", t2-t1)

        return

    def update_main(self):
        self.start_slam_processes()

        # Initialize rendering
        gui.Application.instance.post_to_main_thread(self.window, self._on_start)
        it = iter(self.dataloader)
        timestamp, rgb = next(it)
        self.timestamp = timestamp  # For real-tme handling
        self.idx = 1
        gui.Application.instance.post_to_main_thread(
            self.window, lambda: self.init_render(rgb)
        )

        # Record Data
        # self.timestamps = []
        # self.est_poses = np.array([]).reshape(0, 4, 4)
        self.last_pose = None

        # KF Data
        # kf_buffer_size = 512
        # h, w = self.get_img_size()
        # self.kf_timestamps = -torch.ones(
        #     kf_buffer_size, device=self.device, dtype=torch.double
        # )
        # self.kf_rgb = -torch.ones((kf_buffer_size, 3, h, w), device=self.device)
        # self.kf_depth = -torch.ones((kf_buffer_size, 1, h, w), device=self.device)
        # self.kf_poses = -torch.ones(
        #     (kf_buffer_size, 4, 4), device=self.device, dtype=torch.double
        # )
        # self.P = -torch.ones((0), device=self.device, dtype=torch.double)
        # self.kf_window_end_ind = None

        # Main loop
        fps_interval_len = 30
        start_fps = time.time()
        while not self.tracking_done or not self.mapping_done:
            get_new_frame = True
            if not self.is_running and not self.advance_one_frame:
                get_new_frame = False
                time.sleep(0.01)
                # Update shader rendering if used since user may want to view while paused
                if self.render_val == "Phong":
                    gui.Application.instance.post_to_main_thread(
                        self.window, lambda: self.render_o3d_image()
                    )
            elif not self.is_running and self.advance_one_frame:
                self.advance_one_frame = False

            if get_new_frame:
                # RGB data
                if self.idx < self.dataloader.dataset.__len__():
                    timestamp, rgb = self.load_data(it)
                    self.iter(timestamp, rgb)
                    self.idx += 1
                elif self.idx == self.dataloader.dataset.__len__():
                    # Signal end for exit
                    self.signal_slam_end()
                    time.sleep(5.0)
                    continue

                # Update text
                self.update_idx_text()
                if self.idx % fps_interval_len == 0:
                    end_fps = time.time()
                    elapsed = end_fps - start_fps
                    start_fps = time.time()
                    self.output_fps.text = "FPS: {:.3f}".format(
                        fps_interval_len / elapsed
                    )

        self.shutdown_slam_processes()
        self._on_close()
        gui.Application.instance.quit()

    def setup_slam_processes(self, slam_cfg):
        raise NotImplementedError

    def start_slam_processes(self):
        raise NotImplementedError

    def shutdown_slam_processes(self):
        raise NotImplementedError

    def signal_slam_end(self):
        raise NotImplementedError

    def load_data(self, it):
        raise NotImplementedError

    def iter(self, timestamp, rgb):
        raise NotImplementedError
