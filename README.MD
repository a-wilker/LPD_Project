# LPR Project - Detector de Placas Veiculares

Este projeto é um sistema de visão computacional desenvolvido para a disciplina de Engenharia da Computação. O objetivo é detectar veículos em vídeo, localizar suas placas e extrair os caracteres (OCR) em tempo real.

## 📋 Funcionalidades
- **Detecção de Placas:** Utiliza o modelo YOLOv8.
- **Leitura de Caracteres (OCR):** Integração com EasyOCR e PyTorch (GPU).
- **Base de Dados:** Os registros são salvos automaticamente em `data/registro_placas.csv`.
- **Captura de Evidências:** As imagens das placas detectadas são salvas em `data/output_images`.

## 🛠 Tecnologias Utilizadas
- Python 3.x
- OpenCV (Processamento de Imagem)
- YOLOv8 (Ultralytics)
- EasyOCR
- CUDA (Aceleração por GPU)

## 🚀 Como Rodar
1. Clone o repositório.
2. Instale as dependências:
   ```bash
3. Criar pasta dos dados.

   pip install -r requirements.txt
