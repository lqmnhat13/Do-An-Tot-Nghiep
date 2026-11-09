#!/usr/bin/env bash
# Tải + giải nén dữ liệu đánh giá và tạo overlay thư viện (không sửa env conda).
# Dùng: bash evaluation/benchmarks/setup_data.sh /duong/dan/WORKDIR
# Kết quả: WORKDIR/data/<bộ dữ liệu>, WORKDIR/pyfix (scipy 1.14.1 + pycocotools/h5py/pyarrow).
# Bỏ qua file đã có. Cần mạng; ~4.6 GB.
set -u
W=${1:?Cần thư mục làm việc}
D=$W/data
HF=https://huggingface.co/datasets
PY=${PY:-/opt/anaconda3/envs/ai-macbook/bin/python}
mkdir -p $D/{coco,nyu,tum,vintext,icdar2015,sroie,openvivqa,ktvic}

dl() { [ -s "$1" ] && { echo "SKIP $1"; return; }; curl -sSL --retry 3 -o "$1" "$2" && echo "OK $1" || echo "FAIL $1"; }
dl $D/coco/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip &
dl $D/nyu/val-000000.tar $HF/sayakpaul/nyu_depth_v2/resolve/main/data/val-000000.tar &
dl $D/nyu/val-000001.tar $HF/sayakpaul/nyu_depth_v2/resolve/main/data/val-000001.tar &
dl $D/tum/fr1_desk.tgz https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz &
dl $D/vintext/vintext.zip "https://drive.usercontent.google.com/download?id=1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml&export=download&confirm=t" &
dl $D/icdar2015/ch4_test_images.zip $HF/dlxjj/ICDAR2015/resolve/main/ch4_test_images.zip &
dl $D/icdar2015/Challenge4_Test_Task1_GT.zip $HF/dlxjj/ICDAR2015/resolve/main/Challenge4_Test_Task1_GT.zip &
dl $D/sroie/test.parquet $HF/jsdnrs/ICDAR2019-SROIE/resolve/main/data/test-00000-of-00001.parquet &
dl $D/openvivqa/dev-images.zip $HF/uitnlp/OpenViVQA-dataset/resolve/main/dev-images.zip &
dl $D/openvivqa/vlsp2023_dev_data.json $HF/uitnlp/OpenViVQA-dataset/resolve/main/vlsp2023_dev_data.json &
dl $D/ktvic/test-0.parquet $HF/Skyler215/KTVIC/resolve/main/data/test-00000-of-00006.parquet &
wait

# Bổ sung 2026-11-09: case khó TUM (người đi lại, mặt phẳng trơn sát camera, đi dài trong văn phòng),
# VizWiz (ảnh/câu hỏi thật của người khiếm thị), MC-OCR 2021 (hoá đơn tiếng Việt thật). ~2.9 GB.
mkdir -p $D/vizwiz $D/mcocr
TUM3=https://cvg.cit.tum.de/rgbd/dataset/freiburg3
for s in walking_xyz nostructure_notexture_near_withloop long_office_household; do
  dl $D/tum/fr3_$s.tgz $TUM3/rgbd_dataset_freiburg3_$s.tgz &
done
dl $D/vizwiz/val0.parquet $HF/Multimodal-Fatima/VizWiz_validation/resolve/main/data/validation-00000-of-00005-4ff5ed7be47051bf.parquet &
dl $D/mcocr/val.parquet $HF/DThai/mcocr-test/resolve/main/data/validation-00000-of-00001.parquet &
wait
for s in walking_xyz nostructure_notexture_near_withloop long_office_household; do
  [ -d $D/tum/rgbd_dataset_freiburg3_$s ] || (cd $D/tum && tar xzf fr3_$s.tgz)
done

[ -d $D/vintext/vietnamese ] || (cd $D/vintext && unzip -q vintext.zip)
[ -d $D/openvivqa/dev-images ] || (cd $D/openvivqa && unzip -q dev-images.zip)
[ -d $D/icdar2015/test_images ] || (cd $D/icdar2015 && unzip -q ch4_test_images.zip -d test_images && unzip -q Challenge4_Test_Task1_GT.zip -d test_gt)
[ -d $D/tum/rgbd_dataset_freiburg1_desk ] || (cd $D/tum && tar xzf fr1_desk.tgz)
[ -f $D/vi_docs/labels.json ] || $PY "$(dirname "$0")/make_vi_documents.py" --out $D/vi_docs

# scipy 1.15.3 trong env không nạp được trên macOS 27 (dyld __thread_bss) -> dùng 1.14.1 qua PYTHONPATH.
[ -d $W/pyfix/scipy ] || $PY -m pip install --quiet --no-deps --target $W/pyfix scipy==1.14.1 pycocotools h5py pyarrow
echo "DONE. Chạy với: PYTHONPATH=$W/pyfix HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=1"
