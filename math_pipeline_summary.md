# Handmapping – Báo cáo logic & toán học toàn hệ thống

Tài liệu này mô tả chi tiết toàn bộ pipeline xử lý của dự án Handmapping, từ tín hiệu webcam đến dữ liệu quaternion + motion được phát qua WebSocket và/hoặc retarget vào rig 3D. Nội dung tập trung vào luồng dữ liệu, các biến đổi tọa độ, thuật toán then chốt (Kabsch, SLERP, Lucas–Kanade) và cách chuẩn hóa để đồng bộ với mô hình 3D.

## 1. Tổng quan kiến trúc

| Khối | Nhiệm vụ chính | Sản phẩm |
| --- | --- | --- |
| **`backend/capture.py`** | Cố định webcam ở 1280×720 @30 FPS, lật gương khung hình, trả về tensor BGR. | `frame` (numpy BGR). @backend/capture.py#13-145 |
| **`backend/mp_hands.py`** | Bao MediaPipe HandLandmarker (`detect_for_video`), chuyển BGR→RGB, trả về 21 landmarks chuẩn hóa + handedness đã đảo. | `[{id,x,y,z}]`, `handedness`. @backend/mp_hands.py#102-154 |
| **`backend/pose_solver.py`** | Lưu bind pose, chạy Kabsch cho root rotation, tiện ích quaternion (Euler, SLERP, shortest rotation). | `R`, `quat_root`, `quat_finger`. @backend/pose_solver.py#24-227 |
| **`backend/flow.py`** | Lucas–Kanade optical flow trong pixel space, lấy vận tốc/ góc của từng landmark, làm mượt bằng moving average 5 frame. | `{dx,dy,speed,angle}`. @backend/flow.py#24-175 |
| **`backend/api.py`** | Tracking loop 30 FPS: capture → detect → pose solve → flow → chuẩn hóa packet JSON và broadcast qua WebSocket. | Landmarks, root quaternion, motion, retarget packet. @backend/api.py#161-313 |
| **`backend/retarget_api.py`** | Ánh xạ landmark detector sang hệ rig: mirror tay trái, đổi trục camera→scene, chạy forward kinematics đơn giản và gán quaternion cho từng bone. | `RetargetResult` gồm global points + bone rotations. @backend/retarget_api.py#317-427 |
| **`backend/test_pipeline_3d.py`** | Harness dựng lại tay trong matplotlib 3D, dịch gốc về wrist, hiển thị khung trục kiểm chứng ma trận quay. | Đồ thị 3D để debug tọa độ. @backend/test_pipeline_3d.py#80-165 |

## 2. Chuỗi xử lý của một frame

1. `WebcamCapture.read_frame()` đọc frame BGR, lật ngang (`cv2.flip(frame, 1)`) để hình phản chiếu khớp với tay thật ⇒ gốc tọa độ ảnh vẫn ở góc trên-trái, trục +x sang phải, +y xuống. @backend/capture.py#81-103
2. `HandLandmarker.detect(frame, timestamp_ms)` chuyển frame sang RGB, gọi `detect_for_video` và thu về 21 landmarks chuẩn hóa (`x,y∈[0,1]`, `z` âm khi gần camera hơn cổ tay). Do đã lật ảnh, handedness cần đảo lại. @backend/mp_hands.py#122-154
3. `API.tracking_loop` đóng gói packet landmarks (chuẩn hóa), lần đầu gặp tay sẽ gọi `PoseSolver.set_bind_pose`. @backend/api.py#193-216
4. `PoseSolver.kabsch_rotation` nhận bộ `{wrist, index_mcp, pinky_mcp}` đã trích, căn chỉnh với bind pose để sinh ma trận quay `R` và quaternion, sau đó áp dụng EMA bằng SLERP với hệ số `α`. @backend/api.py#217-249 @backend/pose_solver.py#47-227
5. `OpticalFlowTracker.update` chuyển landmarks chuẩn hóa về pixel, chạy Lucas–Kanade giữa frame trước và frame hiện tại, cộng dồn lịch sử 5 frame để lấy vận tốc mượt. @backend/flow.py#54-175
6. `retarget_api._compute_retarget_packet` (được gọi trong API) dịch landmarks vào to_scene (Y-up), mirror tay trái, cộng delta translation/rotation để ra toạ độ global từng bone, đồng thời tính quaternion ngón tay bằng `_rotation_between`. @backend/api.py#251-257 @backend/retarget_api.py#351-427
7. `WebSocket broadcast` gửi tối đa bốn loại packet: landmarks, root quaternion + Euler, retarget (global points + bone quats) và motion vectors. @backend/api.py#187-297

## 3. Chi tiết toán học & biến đổi

### 3.1 Chuẩn hóa pixel ↔ normalized space

- Điểm ảnh `(u, v)` trong khung hình 1280×720 được chuẩn hóa nội bộ: `x_norm = u / W`, `y_norm = v / H`, với `W=1280`, `H=720`. @backend/mp_hands.py#144-151
- Khi cần tái tạo pixel (ví dụ optical flow), áp dụng `u = x_norm * W`, `v = y_norm * H`. @backend/flow.py#86-114
- Do khung hình đã flip, hệ tọa độ chuẩn hóa vẫn giữ hướng +x sang phải, +y xuống dưới; handedness vì thế phải đổi (`Left ↔ Right`). @backend/mp_hands.py#136-142

### 3.2 Xây dựng không gian quan sát 3D

- Sau khi có landmarks chuẩn hóa, hệ thống dịch tất cả điểm về cổ tay: `p'_i = (x_i - x_w, y_i - y_w, z_i - z_w)`. Cách này đặt gốc tọa độ nội bộ tại wrist, phù hợp để debug hoặc feed vào rig. @backend/test_pipeline_3d.py#91-103
- Bộ tam giác `{wrist (ID0), index_mcp (ID5), pinky_mcp (ID17)}` đại diện cho mặt phẳng lòng bàn tay. Các vector này được chuyển thành `numpy.array` để đưa vào Kabsch. @backend/mp_hands.py#242-271

#### 3.2.1 Pipeline chi tiết ảnh 2D → tọa độ 3D

1. **Mẫu hóa pixel từ webcam**  
   - `WebcamCapture.read_frame()` trả tensor BGR kích thước `(H=720, W=1280, 3)` sau khi lật gương, giữ hệ tọa độ ảnh với gốc (0,0) ở góc trên-trái. @backend/capture.py#81-103  
   - Mỗi pixel `(u, v)` được lưu với giá trị 8-bit, chuẩn bị cho các bước chuyển đổi tiếp theo.

2. **Chuẩn hóa sang không gian MediaPipe**  
   - `HandLandmarker.detect` chuyển BGR→RGB rồi gói thành `mp.Image`. MediaPipe yêu cầu tọa độ chuẩn hóa nên mỗi điểm ảnh được nội suy thành `x_norm = u / W`, `y_norm = v / H`. @backend/mp_hands.py#122-151  
   - Nhờ chuẩn hóa, mô hình học sâu trở nên độc lập với độ phân giải tuyệt đối, chỉ phụ thuộc tỉ lệ hình.

3. **Suy luận độ sâu tương đối bằng MediaPipe**  
   - Hàm `detect_for_video` chạy mạng Hand Landmarker và trả về 21 điểm `(x_norm, y_norm, z_rel)` với `z_rel` là khoảng cách chuẩn hóa so với cổ tay; giá trị âm nghĩa là điểm gần camera hơn. @backend/mp_hands.py#129-151  
   - MediaPipe không đo chiều sâu tuyệt đối mà ước lượng dựa trên hình học bàn tay và kích thước mẫu đã được huấn luyện.

4. **Chuyển ngược về pixel/đơn vị thực**  
   - Khi cần truy hồi về pixel (ví dụ optical flow), áp dụng `u = x_norm · W`, `v = y_norm · H`. `z_rel` có thể được scale bởi một hệ số kinh nghiệm (ví dụ chiều dài xương cổ tay) để tạo nên đơn vị mm tương đối. @backend/flow.py#86-114  
   - Đối với debug 3D, dự án giữ giá trị `z_rel` nguyên bản nhằm bảo toàn quan hệ hình học tương đối giữa các ngón tay.

5. **Đặt gốc tại cổ tay & tạo không gian camera 3D**  
   - Vector cổ tay `p_w = (x_w, y_w, z_w)` được trừ khỏi từng landmark: `p'_i = p_i - p_w`, biến toàn bộ dữ liệu sang hệ tọa độ cục bộ đặt tại wrist, giúp loại bỏ thành phần dịch chuyển tuyệt đối trong ảnh. @backend/test_pipeline_3d.py#91-103  
   - Bộ điểm `p'_i` chính là “không gian 3D quan sát” (camera space) mà các bước sau sẽ sử dụng.

6. **Căn chỉnh sang bind pose & world space**  
   - Từ `p'_i`, trích `{W,I,P}` để chạy Kabsch ⇒ thu ma trận quay `R` của toàn bàn tay. @backend/pose_solver.py#69-109  
   - Khi cần đưa vào scene Y-up hoặc mirror tay trái, áp dụng ma trận đổi trục `to_scene = diag([1,-1,1])` và `hand_matrix = diag([-1,1,1])` (nếu handedness = Left). @backend/retarget_api.py#354-401  
   - Kết quả là bộ tọa độ 3D đã ở hệ rig/engine, sẵn sàng để áp dụng cho skeleton hoặc dùng làm dữ liệu debug.

#### 3.2.2 Danh sách 21 landmarks & vai trò

MediaPipe Hands luôn trả về 21 điểm `id ∈ [0,20]` với thứ tự cố định. Dự án sử dụng đầy đủ các điểm này cho Kabsch, optical flow và retarget từng ngón. Bảng dưới tóm tắt tên khớp, mô tả sinh học và vai trò trong pipeline:

| ID | Tên khớp (MediaPipe) | Mô tả & vị trí | Vai trò chính |
| --- | --- | --- | --- |
| 0 | Wrist | Gốc cổ tay | Gốc mọi phép dịch; điểm tham chiếu để trừ tọa độ và đặt bind pose. |
| 1 | Thumb_CMC | Khớp gốc ngón cái (carpometacarpal) | Mốc đầu chuỗi ngón cái khi tính vector xương. |
| 2 | Thumb_MCP | Khớp MCP ngón cái | Dùng cho quaternion `_rotation_between` của xương thumb proximal. |
| 3 | Thumb_IP | Khớp IP ngón cái | Trung gian trước fingertip, tham gia tính motion và retarget. |
| 4 | Thumb_TIP | Đầu ngón cái | Theo dõi gesture (pinch, swipe) dựa trên motion speed. |
| 5 | Index_MCP | Khớp MCP ngón trỏ | Một trong ba điểm chạy Kabsch (cùng wrist/pinky). |
| 6 | Index_PIP | Khớp PIP ngón trỏ | Vector MCP→PIP dùng tính quaternion finger. |
| 7 | Index_DIP | Khớp DIP ngón trỏ | Trung gian trước fingertip. |
| 8 | Index_TIP | Đầu ngón trỏ | Theo dõi tương tác, gesture, motion. |
| 9 | Middle_MCP | Khớp MCP ngón giữa | Mốc để tạo tam giác lòng bàn tay phụ (debug). |
| 10 | Middle_PIP | Khớp PIP ngón giữa | Thành phần vector finger chain. |
| 11 | Middle_DIP | Khớp DIP ngón giữa | Như trên. |
| 12 | Middle_TIP | Đầu ngón giữa | Theo dõi motion và retarget chain. |
| 13 | Ring_MCP | Khớp MCP ngón áp út | Nằm trên vòng cung lòng bàn tay, hỗ trợ ổn định mesh. |
| 14 | Ring_PIP | Khớp PIP ngón áp út | Vector finger chain. |
| 15 | Ring_DIP | Khớp DIP ngón áp út | Vector finger chain. |
| 16 | Ring_TIP | Đầu ngón áp út | Motion + retarget. |
| 17 | Pinky_MCP | Khớp MCP ngón út | Điểm thứ ba trong bộ tam giác Kabsch. |
| 18 | Pinky_PIP | Khớp PIP ngón út | Vector finger chain. |
| 19 | Pinky_DIP | Khớp DIP ngón út | Vector finger chain. |
| 20 | Pinky_TIP | Đầu ngón út | Motion + retarget. |

**Những điểm trọng yếu**  
- `{0,5,17}`: bộ tam giác xác định mặt phẳng lòng bàn tay và được PoseSolver dùng cho Kabsch. @backend/mp_hands.py#242-271  
- `{4,8,12,16,20}`: các fingertip được OpticalFlowTracker ưu tiên log tốc độ/góc để làm gesture recognition. @backend/flow.py#246-323  
- Các cạnh liên tiếp trong từng ngón (ví dụ `(5,6),(6,7),(7,8)`) được retarget API quét để tính quaternion từng xương bằng `_rotation_between`. @backend/retarget_api.py#369-426

### 3.3 Thuật toán Kabsch cho root rotation

1. Tạo hai ma trận `P` (bind pose) và `Q` (observed) với mỗi hàng là tọa độ của `{W, I, P}`. @backend/pose_solver.py#69-80
2. Trừ centroid: `P_c = P - mean(P)`, `Q_c = Q - mean(Q)`. Việc này đảm bảo phép quay diễn ra quanh trọng tâm tam giác, tương đương root bone. @backend/pose_solver.py#82-88
3. Tính `H = P_c^T Q_c`, chạy SVD `H = U S V^T`, sau đó `R = V U^T`. Nếu `det(R) < 0`, đảo cột cuối của `V` để loại reflexion. @backend/pose_solver.py#89-103
4. Chuyển `R` sang quaternion `[x, y, z, w]` bằng `scipy.spatial.transform.Rotation`. @backend/pose_solver.py#104-107
5. Trong `API.tracking_loop`, quaternion mới `q_new` được trộn với `q_prev` bằng SLERP: `q = slerp(q_prev, q_new, α)` với `α = ema_alpha`. Nếu dot < 0 thì đảo dấu `q2` để đi đường ngắn nhất. @backend/api.py#223-230 @backend/pose_solver.py#190-226

### 3.4 Euler & hệ trục

- Quaternion sau khi smoothing được chuyển sang Euler `xyz` (roll/pitch/yaw, đơn vị độ) cho mục đích debug/truyền thông tin phụ. @backend/api.py#232-248 @backend/pose_solver.py#123-135
- Hệ Camera/MediaPipe mặc định `+x` sang phải, `+y` xuống, `+z` ra khỏi camera (giá trị âm tiến gần camera). Nếu rig dùng `+Y` lên, cần nhân thêm ma trận đổi trục `R_swap = diag(1,-1,1)` trước khi convert sang quaternion. `test_pipeline_3d` hiện vẽ trực tiếp hệ Camera nên thấy tay bị lật nếu không đổi trục. @backend/test_pipeline_3d.py#147-176

### 3.5 Optical flow & động học tuyến tính

- Lucas–Kanade tính toán chuyển động điểm giữa hai frame gray-scale dùng cửa sổ 21×21, pyramid maxLevel=3. Kết quả là cặp tọa độ `next_points` và trạng thái `status`. @backend/flow.py#31-115
- Vector chuyển động: `dx = next_x - prev_x`, `dy = next_y - prev_y`. Hệ thống lưu deque 5 phần tử/landmark để tính trung bình trượt: 
  \[
  \bar{dx}_i = \frac{1}{N}\sum_{k=1}^N dx_{i,k},\quad \bar{dy}_i = \frac{1}{N}\sum_{k=1}^N dy_{i,k}
  \]
  với `N ≤ 5`. @backend/flow.py#129-150
- Tốc độ và góc: `speed = sqrt(\bar{dx}^2 + \bar{dy}^2)`, `angle = atan2(\bar{dy}, \bar{dx})` (độ). Đây là dữ liệu đầu vào cho các thuật toán gesture/velocity downstream. @backend/flow.py#139-150
- Nếu tracking thất bại (`status=0`), giá trị trung bình gần nhất được dùng để tránh nhảy số. @backend/flow.py#151-169

### 3.6 Retarget sang rig 3D

1. Chuẩn bị: `compute_global_points` nhận landmarks detector, pose hiện tại của model, bảng mapping landmark→bone và delta transform (dịch + quay) trong global space. @backend/retarget_api.py#317-333
2. Gương tay trái: nếu handedness = Left, áp dụng ma trận mirror `diag([-1,1,1])` trước và sau ma trận quay để đổi chỗ trục X. @backend/retarget_api.py#354-359
3. Đổi trục camera→scene: `to_scene = diag([1,-1,1])` để đưa từ hệ MediaPipe (Y xuống) sang Three.js (Y lên). @backend/retarget_api.py#360-383
4. Forward kinematics đơn giản: `forward_kinematics_from_landmarks` sử dụng chuỗi ngón tay [(1,2),(2,3),(3,4),…] để dựng local points cho từng bone, sau đó cộng delta: `world_pos = R_delta @ local + d_pos`. @backend/retarget_api.py#369-391
5. Quaternion ngón tay: mỗi cạnh MCP→PIP→DIP→TIP tạo vector `direction = child - parent`. Sau khi đổi trục/nhân mirror, vector chuẩn hóa và dùng `_rotation_between` để quay từ hướng chuẩn `bone_dir=[0,1,0]` sang hướng mục tiêu. @backend/retarget_api.py#403-426
6. Kết quả trả về `RetargetResult(global_points, bone_rotations)` để client 3D áp dụng trực tiếp.

## 4. Không gian tọa độ & phép biến đổi

| Không gian | Định nghĩa gốc & trục | Phép chuyển quan trọng | Ghi chú |
| --- | --- | --- | --- |
| **Pixel (Webcam)** | (0,0) góc trên-trái, +x sang phải, +y xuống; ảnh đã flip ngang. | `frame = cv2.flip(raw_frame, 1)` để đồng nhất với tay người dùng. @backend/capture.py#93-100 | Input cho MediaPipe & optical flow. |
| **Normalized (MediaPipe)** | Giữ gốc trên-trái, `z` âm hướng camera, tọa độ ∈[0,1]. | `u = x_norm·W`, `v = y_norm·H` khi cần quay lại pixel. @backend/mp_hands.py#144-151 @backend/flow.py#86-114 | Cấp dữ liệu gốc cho mọi mô-đun. |
| **Observational 3D** | Dịch wrist về (0,0,0); trục giống normalized. | `p'_i = p_i - p_wrist`. @backend/test_pipeline_3d.py#91-103 | Dùng cho debug, rig offline. |
| **Bind pose space** | Định bởi `PoseSolver.set_bind_pose` với tam giác {W,I,P}. | Kabsch căn chỉnh observed vào bind ⇒ root rotation. @backend/pose_solver.py#24-107 | Ràng buộc rig với dữ liệu thực. |
| **Scene/model space** | Hệ toạ độ engine (Three.js, Unity…). | `to_scene = diag([1,-1,1])`, mirror tay trái, thêm delta translation. @backend/retarget_api.py#354-391 | Dữ liệu cuối để render. |

### 4.1 Các hệ không gian trong dự án

#### Pixel/Webcam Space
- **Định nghĩa**: ma trận ảnh BGR 1280×720 sau `cv2.flip(frame, 1)`, gốc (0,0) góc trên-trái; +x sang phải, +y xuống. @backend/capture.py#81-103  
- **Vai trò**: nguồn dữ liệu cho MediaPipe và optical flow; mọi thao tác vẽ/đọc pixel diễn ra tại đây. @backend/flow.py#54-175  
- **Biến đổi chính**: chuyển BGR→RGB, chuẩn hóa `(u/W, v/H)`, hoặc giữ nguyên pixel để tính Lucas–Kanade.

#### MediaPipe Normalized Space
- **Định nghĩa**: `(x,y)∈[0,1]`, `z_rel` theo chuẩn camera của MediaPipe (x phải, y xuống, z âm). @backend/mp_hands.py#122-154  
- **Vai trò**: hệ quy chiếu chung cho toàn pipeline (pose solver, retarget, motion). Không cần biết ma trận nội tại camera.  
- **Biến đổi chính**: chuẩn hóa từ pixel (`x=u/W`, `y=v/H`), đảo handedness sau khi lật ảnh, chuyển ngược về pixel khi cần (`u=x·W`, `v=y·H`).

#### Wrist-centered Observational 3D
- **Định nghĩa**: tọa độ normalized bị trừ vector cổ tay `p'_i = p_i - p_w`, gốc đặt tại wrist, trục giữ nguyên chuẩn MediaPipe. @backend/test_pipeline_3d.py#91-103  
- **Vai trò**: loại bỏ dịch chuyển tuyệt đối, giữ lại hình dáng/tư thế để vẽ 3D và chạy Kabsch.  
- **Biến đổi chính**: sử dụng trực tiếp trong `test_pipeline_3d` và `pose_solver`; không scale thêm.

#### Bind Pose Space
- **Định nghĩa**: snapshot `{wrist, index_mcp, pinky_mcp}` thu từ frame đầu tiên, lưu trong `PoseSolver.bind_pose`. @backend/pose_solver.py#24-46  
- **Vai trò**: mốc so sánh cho Kabsch; đại diện cho tư thế neutral của rig.  
- **Biến đổi chính**: so sánh tam giác observed với tam giác bind, trừ centroid rồi chạy SVD để lấy `R` và `quat`.

#### Scene/Model Space
- **Định nghĩa**: hệ tọa độ engine (Three.js, Unity…) với gốc tại root bone của rig, thường `+Y` lên, `+Z` ra trước.  
- **Vai trò**: nơi áp dụng quaternion lên xương thật. `retarget_api` đã đổi trục (`to_scene`), mirror tay trái (`hand_matrix`) và cộng delta translation/rotation của rig. @backend/retarget_api.py#354-401  
- **Biến đổi chính**: `world_pos = rot_matrix @ (to_scene @ hand_matrix @ local) + delta_pos`, đồng thời sinh `bone_rotations` bằng `_rotation_between`. @backend/retarget_api.py#369-426

#### Motion/Velocity Space
- **Định nghĩa**: không gian vectơ chuyển động trong pixel, được tính bởi optical flow và lưu dưới dạng `(dx, dy, speed, angle_deg)` cho mỗi landmark. @backend/flow.py#54-175  
- **Vai trò**: feed vào các hệ gesture hoặc animation blending, độc lập với các phép quay Kabsch.  
- **Biến đổi chính**: Lucas–Kanade trên ảnh gray-scale, moving average 5 frame, chuyển đổi tốc độ/góc để broadcast qua WebSocket.

## 5. Công thức chủ chốt

1. **Chuẩn hóa & dịch gốc**  
   - `x_norm = u / W`, `y_norm = v / H`.  
   - `p'_i = p_i - p_wrist`.  
2. **Kabsch**  
   - `H = P_c^T Q_c`, `U S V^T = SVD(H)`, `R = V U^T`, `quat = as_quat(R)`.  
3. **SLERP (EMA)**  
   - `q = sin((1-t)θ)/sinθ * q1 + sin(tθ)/sinθ * q2`, với `θ = arccos(dot(q1,q2))`.  
4. **Shortest rotation cho ngón**  
   - `axis = normalize(v_from × v_to)`, `angle = arccos(dot)`, `quat = [axis·sin(angle/2), cos(angle/2)]`. @backend/pose_solver.py#136-188
5. **Optical flow velocity**  
   - `v = (\bar{dx}, \bar{dy})`, `speed = ||v||`, `angle = atan2(\bar{dy}, \bar{dx})`. @backend/flow.py#139-150
6. **Retarget chuyển toạ độ**  
   - `rot_matrix = euler_deg_to_matrix(delta.drot_deg)`; nếu tay trái: `R = M R M`.  
   - `world_pos = rot_matrix @ (to_scene @ hand_matrix @ local) + delta_pos`. @backend/retarget_api.py#351-400

## 6. Chuẩn I/O & schema packet

| Hàm/Module | Input | Output | Lưu ý |
| --- | --- | --- | --- |
| `HandLandmarker.detect(frame, t)` | Frame BGR, timestamp | `(landmarks, handedness)` với `landmarks[i]={id,x,y,z}` | Chỉ trả về tay đầu tiên (num_hands=1). @backend/mp_hands.py#102-154 |
| `PoseSolver.kabsch_rotation(key_lms)` | Dict `{wrist,index_mcp,pinky_mcp}` | `(R 3×3, quat [x,y,z,w])` | Yêu cầu đã gọi `set_bind_pose`. @backend/pose_solver.py#47-109 |
| `PoseSolver.shortest_rotation_quat(v_from, v_to)` | Hai vector 3D | Quaternion quay ngắn nhất | Dùng cho mỗi đốt ngón. @backend/pose_solver.py#136-188 |
| `OpticalFlowTracker.update(frame, lms)` | Frame BGR, landmarks chuẩn hóa | Danh sách `{id,dx,dy,speed,angle_deg}` | Khởi tạo trả về zero motion. @backend/flow.py#86-175 |
| `API.tracking_loop` | None (loop) | Broadcast 1..4 packet JSON | Landmarks, root quaternion, retarget, motion. @backend/api.py#187-297 |
| `compute_global_points` | Detector landmarks, rig pose, mapping, delta | `RetargetResult` (global_points, bone_rotations) | Mirror tay trái + đổi trục + FK. @backend/retarget_api.py#317-427 |

## 7. Lộ trình thao tác dữ liệu 3D → rig

> Mục tiêu: chuyển từ tập landmark chuẩn hóa (camera space) sang pose đầy đủ cho rig 3D (scene space) với root orientation, tỉ lệ xương và quaternion cho từng khớp. Các bước dưới đây mô tả logic, công thức và cấu trúc dữ liệu đi kèm.

1. **Thiết lập bind pose khớp rig**  
   - Thu bộ `{W, I, P}` (ID 0, 5, 17) ở frame đầu tiên, truyền vào `PoseSolver.set_bind_pose`. Hàm này lưu `self.bind_pose = { 'wrist': np.array(...), ... }` để làm mốc so sánh. @backend/pose_solver.py#24-46  
   - Rig cũng phải có “tam giác bind” tương ứng (ví dụ root bone + `index_mcp` + `pinky_mcp`). Quy trình khuyến nghị: (a) đưa rig về tư thế neutral, (b) đảm bảo camera nhìn đúng pose này, (c) gọi `set_bind_pose` để lưu lại. Như vậy `Kabsch(P_obs, P_bind)` sẽ trả quaternion đúng với root của rig.  
   - Nếu cần cập nhật bind (ví dụ rig mới), chỉ cần reset `PoseSolver` và thiết lập lại khi phát hiện tay đầu tiên.

2. **Chuẩn hóa theo wrist & tỉ lệ**  
   - Server luôn truyền landmark đã trừ wrist: `p'_i = p_i - p_wrist`. Điều này tương đương đặt gốc toạ độ tại cổ tay, giúp root bone của rig (thường là wrist) nhận dữ liệu dịch chuyển = 0. @backend/test_pipeline_3d.py#91-145  
   - Tỉ lệ xương được khôi phục bằng hệ số: 
     \[
     S_{b} = \frac{\lVert p^{\text{rig}}_{\text{child}} - p^{\text{rig}}_{\text{parent}}\rVert}{\lVert p^{\text{obs}}_{\text{child}} - p^{\text{obs}}_{\text{parent}}\rVert}
     \]
     Áp dụng riêng cho từng cạnh để tránh sai số tích luỹ. Trong thực tế, có thể đo `S_b` từ chính lần bind đầu tiên rồi lưu lại cho client.  
   - Khi dựng lại mesh hoặc tính FK, nhân mỗi vector quan sát với `S_b` tương ứng trước khi cộng dồn chiều dài xương.

3. **Áp dụng root rotation**  
   - Với mỗi frame, `PoseSolver.kabsch_rotation` nhận `key_lms = {'wrist': ..., 'index_mcp': ..., 'pinky_mcp': ...}` và trả về `(R, q_root)`. Công thức: `R = VU^T` sau khi SVD `H = P_c^T Q_c`, sau đó `quat = as_quat(R)`. @backend/pose_solver.py#69-109  
   - `API.tracking_loop` lưu `prev_root_quat` và nội suy SLERP: `q_t = slerp(q_{t-1}, q_root, α)` để triệt tiêu nhiễu cao tần. Euler `xyz` cũng được tính kèm để debug. @backend/api.py#217-249  
   - Đổi trục: nếu engine sử dụng chuẩn `+Y lên`, áp dụng `R_swap = diag(1,-1,1)` trước khi chuyển sang quaternion cuối. Khi phát retarget packet, hàm `compute_global_points` đã nhân `to_scene = diag([1,-1,1])` và `hand_matrix = diag([-1,1,1])` cho tay trái nên client chỉ cần đọc dữ liệu nếu chung chuẩn. @backend/retarget_api.py#354-401

4. **Tính rotation từng khớp**  
   - **Thuật toán tổng quát**: 
     1. Lấy vector bind `v_{bind} = normalize(p^{bind}_{child} - p^{bind}_{parent})`.  
     2. Lấy vector quan sát `v_{obs} = normalize(p'_{child} - p'_{parent})`.  
     3. Dùng `shortest_rotation_quat(v_{bind}, v_{obs})` để thu quaternion local. @backend/pose_solver.py#136-188  
     4. Nếu hệ toạ độ khác, đổi trục cho cả hai vector trước khi tính.  
     5. Với tay trái, nhân thêm `hand_matrix` để giữ đúng hướng ngón.  
   - **Hiện thực có sẵn**: `forward_kinematics_from_landmarks` (retarget API) duyệt các chuỗi ngón: Thumb [(1,2),(2,3),(3,4)], Index [(5,6),(6,7),(7,8)], …; tính vector `direction = child - parent`, đổi trục (`to_scene`), mirror nếu `handedness = left`, chuẩn hoá rồi gọi `_rotation_between(bone_dir, target_dir)` với `bone_dir = [0,1,0]`. Kết quả quaternion lưu trong `RetargetResult.bone_rotations`. @backend/retarget_api.py#369-426

5. **Đồng bộ thời gian thực & packet schema**  
   - Chuỗi broadcast: 
     ```text
     frame -> HandLandmarker -> PoseSolver (bind + kabsch + slerp) -> OpticalFlow -> RetargetAPI
          -> WebSocket packets {landmarks | root_quaternion | retarget | motion}
     ```  
   - Trường dữ liệu chính:
     - `root_quaternion`: `{quaternion: {x,y,z,w}, euler_deg: {roll,pitch,yaw}}`.  
     - `retarget`: `{global_points: {bone: [x,y,z]}, bone_rotations: {bone: [x,y,z,w]}}`.  
     - `motion`: `{id, dx, dy, speed, angle_deg}` phục vụ gesture detection. @backend/api.py#187-297  
   - Client-side xử lý khuyến nghị: 
     1. Lưu `bind_pose_client` và mapping bone↔landmark giống server. 
     2. Khi nhận packet mới: 
        - Cập nhật root orientation bằng `root_quaternion`. 
        - Áp dụng `retarget.bone_rotations` trực tiếp (hoặc tự tính theo công thức mục 4 để double-check). 
        - Sử dụng `global_points` để đặt vị trí joint hoặc vẽ debug gizmo. 
        - Dựa vào `motion` để blend animation (ví dụ khi `speed` vượt ngưỡng thì chuyển sang state “wave”). 
     3. Nếu rig chạy ở tần số khác (ví dụ 60 FPS), nội suy quaternion giữa hai packet bằng SLERP để giữ mượt.
