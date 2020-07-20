import torch
import numpy as np
import librosa

from vasync.datasets.vggsound.vggish import VGGish
from vasync.datasets.vggsound import vggish_input, vggish_postprocess


class VGGFeatExtractor():
    def __init__(self):
        # Initialize the PyTorch model.
        self.device = 'cuda:0'
        pytorch_model = VGGish()
        pytorch_model.load_state_dict(torch.load('vasync/datasets/vggsound/pytorch_vggish.pth'))
        self.pytorch_model = pytorch_model.to(self.device)
        self.pytorch_model.eval()
        self.post_processor = vggish_postprocess.Postprocessor('vasync/datasets/vggsound/vggish_pca_params.npz')

    def cal_feat(self, input_batch):
        pytorch_output = self.pytorch_model(input_batch)
        pytorch_output = pytorch_output.detach().cpu().numpy()
        return pytorch_output
    
    def get_emb(self, wav_path):
        # Generate a sample input (as in the AudioSet repo smoke test).
        num_secs = 5
        x, sr = librosa.load(wav_path)
        x = x[:sr*num_secs]
        # Produce a batch of log mel spectrogram examples.
        input_batch = vggish_input.waveform_to_examples(x, sr)
        input_batch = torch.from_numpy(input_batch).unsqueeze(dim=1)
        input_batch = input_batch.float().to(self.device)
        # Run the PyTorch model.
        pytorch_output = self.cal_feat(input_batch)
        postprocessed_output = self.post_processor.postprocess(pytorch_output)
        
        return pytorch_output, postprocessed_output


def main():
    vgg_feat_extractor = VGGFeatExtractor()
    
    wav_path = "/raid/hhemati/Datasets/MultiModal/VGGSound/wavs/video_12512.wav"
    pytorch_output, postprocessed_output = vgg_feat_extractor.get_emb(wav_path)

    # print(pytorch_output[-1])
    # print(postprocessed_output[-1])
    # print(postprocessed_output[-1].shape)
    expected_embedding_mean = 0.131
    expected_embedding_std = 0.238
    print('Computed Embedding Mean and Standard Deviation:', np.mean(pytorch_output), np.std(pytorch_output))
    print('Expected Embedding Mean and Standard Deviation:', expected_embedding_mean, expected_embedding_std)

    # Post-processing.
    post_processor = vggish_postprocess.Postprocessor('vasync/datasets/vggsound/vggish_pca_params.npz')
    postprocessed_output = post_processor.postprocess(pytorch_output)
    expected_postprocessed_mean = 123.0
    expected_postprocessed_std = 75.0
    print('Computed Post-processed Embedding Mean and Standard Deviation:', np.mean(postprocessed_output),
          np.std(postprocessed_output))
    print('Expected Post-processed Embedding Mean and Standard Deviation:', expected_postprocessed_mean,
          expected_postprocessed_std)


if __name__ == '__main__':
    main()