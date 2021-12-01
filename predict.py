# Import libraries
import torch

import preprocess as pr


class PredictStock:
    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, data):
        # Load Model
        model = torch.load(self.model_path)
        model.eval()

        # Preprocess Data
        x_columns = ['ISE.1', 'SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU']
        x = data[x_columns].values
        x = torch.from_numpy(x).float()
        x = x.reshape(1, x.size(0), x.size(1))

        # Predict
        with torch.no_grad():
            pred, _ = model(x)

        return pred[0][-1].item()


if __name__ == '__main__':
    data_pth = './data/user_data.csv'
    model_pth = 'data/best_model.pt'
    df = pr.load_data(data_pth)
    out = PredictStock(model_pth).predict(df)
    print(out)
