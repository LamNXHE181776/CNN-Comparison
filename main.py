import os
from train_cnn import Trainer
import seaborn as sns
from models.CNN import CNN
from models.ResNet import ResNet
from torch import nn
import time

sns.set_style("whitegrid")

def main():
    funcs = [nn.ReLU, nn.SiLU, nn.LeakyReLU, nn.GELU]
    model1 = CNN
    p = model1.__name__
    cnn_trainer = Trainer(model_type=model1, act_func=nn.SiLU, epochs = 120)
    summary = cnn_trainer.model_summary()
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    k = cnn_trainer.act_func.__name__

    os.makedirs(f"results/{p}/{k}/{current_time}/", exist_ok=True)
    with open(f"results/{p}/{k}/{current_time}/{p}_model_summary.txt", "w") as f:
        f.write(str(summary))

    # for func in funcs:
    #   cnn_trainer = Trainer(model_type=CNN, act_func=nn.ReLU, epochs=30)
    
    

    print(f"Using activation function: {k}")
    cnn_trainer.train_model(save_path=f"./saved_models/{p}/{k}/{current_time}")
    cnn_trainer.plot_training_results(plot_path=f"./results/{p}/{k}/{current_time}")
    cnn_trainer.test_model(model_path=f"./saved_models/{p}/{k}/{current_time}/best_fashion_model.pth")
    cnn_trainer.test_model(model_path=f"./saved_models/{p}/{k}/{current_time}/final_fashion_model.pth")
    

if __name__ == "__main__":
    main()    