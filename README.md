# adversarial_attack
- using advertorch library
https://github.com/BorealisAI/advertorch

### command
1. Training
  - normal model
    ```
    python main.py --model=<model type> --n_way=<number of classes> --imgsz=<image size>
    ```
  - robust model
    ```
    python main.py --model=<model type> --n_way=<number of classes> --imgsz=<image size> --train_attack=<attack name> --train_eps=<attack bound>
    ```

2. Make attacked image
    ```
    python main.py --pretrained=<pretrained model path> --imgsz=<image size> --test_attack=<attack name> --test_eps=<attack bound>
    ```
  

