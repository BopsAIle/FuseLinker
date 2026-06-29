# FuseLinker

FuseLinker là một dự án liên quan đến link prediction trên knowledge graph bằng cách kết hợp nhiều nguồn thông tin, bao gồm:
- embedding văn bản (text embeddings)
- embedding tri thức/domain knowledge
- mô hình graph neural network (R-GCN)
- hỗ trợ PPR-based auxiliary graph cho các biến thể nâng cao

Dự án được thiết kế để huấn luyện và đánh giá các mô hình trên các bộ dữ liệu như `hetionet`, `suppkg`, `kegg50k` và `adint`.

## Tính năng chính

- Huấn luyện mô hình link prediction trên các tập dữ liệu khác nhau
- Tính AUROC và vẽ đồ thị ROC cho checkpoint đã huấn luyện
- So sánh ROC giữa các biến thể mô hình (base/upgrade)
- Hỗ trợ nhiều loại embedding văn bản như `pubmedbert`, `flant5`, `llama2`, `pmcllama`, `bert`

## Cấu trúc thư mục chính

```text
fuselinker/
  main.py                # huấn luyện mô hình chính
  main2.py               # huấn luyện biến thể cơ bản
  eval_auroc.py          # đánh giá AUROC từ checkpoint
  plot_roc_comparision.py  # vẽ so sánh ROC trên nhiều dataset
  data_loader.py
  model.py
  model_base2.py
  model_base4.py
  checkpoints/           # lưu checkpoint và ROC plots
  hetionet/              # dữ liệu hetionet
  suppkg/                # dữ liệu suppkg
  kegg50k/               # dữ liệu kegg50k
  adint/                 # dữ liệu adint
```

## Yêu cầu môi trường

- Python 3.10+
- PyTorch
- DGL
- pandas / numpy / matplotlib

Cài đặt phụ thuộc:

```bash
cd /home/anhlq/Documents/2026-ltt/FuseLinker
python -m pip install -r requirements.txt
```

Nếu đang dùng môi trường ảo, có thể kích hoạt trước khi chạy:

```bash
cd /home/anhlq/Documents/2026-ltt/FuseLinker
source .venv/bin/activate
```

## Chuẩn bị dữ liệu và embedding

Mỗi bộ dữ liệu cần có các file:
- `train.tsv`
- `valid.tsv`
- `test.tsv`
- `poincare_embeddings.npy`
- file embedding văn bản tương ứng (ví dụ `pubmedbert_pretrained_embeddings_768.npy`)

Một số embedding có thể được tải từ đường link bên dưới và đặt vào thư mục dataset tương ứng trước khi chạy:

- https://drive.google.com/drive/folders/1aIsdgX7IUqMl4Wn3TFEU-1iAU0kVfBpu?usp=sharing

## Huấn luyện mô hình

Chạy từ thư mục `fuselinker/`:

```bash
cd /home/anhlq/Documents/2026-ltt/FuseLinker/fuselinker
```

Ví dụ huấn luyện trên `hetionet`:

```bash
python main.py \
  --data hetionet \
  --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
  --knowledge_embedding_file poincare_embeddings.npy \
  --num_hidden_layers 2 \
  --iterations 40000 \
  --evaluate_every 1000 \
  --neg_sample_size_eval 100 \
  --w 0.75 \
  --model_state_file checkpoints/hetionet/pubmedbert/model/model_state.pth
```

Ví dụ huấn luyện biến thể cơ bản (không dùng PPR):

```bash
python main2.py \
  --data hetionet \
  --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
  --knowledge_embedding_file poincare_embeddings.npy \
  --num_hidden_layers 2 \
  --iterations 40000 \
  --evaluate_every 1000 \
  --neg_sample_size_eval 100 \
  --w 0.75 \
  --model_state_file checkpoints/hetionet/pubmedbert/model_state.pth
```

## Đánh giá AUROC từ checkpoint

Sau khi đã có checkpoint, có thể đánh giá trực tiếp mà không cần huấn luyện lại:

```bash
python eval_auroc.py \
  --data hetionet \
  --text_embedding_file pmcllama_pretrained_embeddings_4096.npy \
  --knowledge_embedding_file poincare_embeddings.npy \
  --num_hidden_layers 2 \
  --w 0.75 \
  --model_state_file checkpoints/hetionet/pmcllama/model/model_state.pth
```

Kết quả sẽ gồm:
- AUROC in ra terminal
- file ROC plot được lưu cùng thư mục checkpoint

## Vẽ đồ thị ROC so sánh nhiều dataset

Để so sánh ROC giữa các dataset như `hetionet`, `suppkg`, `kegg50k`:

```bash
python plot_roc_comparision.py \
  --root . \
  --datasets hetionet suppkg kegg50k \
  --output roc_comparison.png \
  --skip-missing
```

File ảnh đầu ra sẽ được lưu tại `fuselinker/roc_comparison.png`.
## Check point 
https://drive.google.com/drive/folders/1JMCVMFY9y_iLxA2UONq78XqD64OG_L0N
## Ghi chú

- Nếu checkpoint không tồn tại, hãy kiểm tra lại đường dẫn trong `--model_state_file`.
- Nếu dùng embedding khác, hãy đổi đúng tên file trong tham số `--text_embedding_file`.
- Một số biến thể mô hình có thể nằm ở các thư mục con khác nhau như `model/`, `modelbase/`, `modelbase2/` trong `checkpoints/`.
