# Enhancing Character-Level Recurrent Neural Networks for Text Generation

This repository contains the code and analysis for the project titled "Enhancing Character-Level Recurrent Neural Networks for Text Generation: A Comparative Analysis on LSTM Architectures and Data Augmentation". This project was submitted as part of the COGS 185 Advanced Machine Learning Methods course at the University of California San Diego.

## Abstract
In this project, I explore the effectiveness of character-level recurrent neural networks (Char RNN) for text generation. The study focuses on improving model performance through alterations in Long Short-Term Memory (LSTM) architectures, including comparing single-layer and double-layer LSTMs, then comparing the superior model with and without data augmentation and regularization techniques. The Tiny Shakespeare dataset was used to evaluate the performance of the models through key metrics such as loss, perplexity, and accuracy.

## Methodology
### Data Preprocessing
The Tiny Shakespeare dataset was used for analysis, which includes 40,000 lines of text from Shakespeare plays, totaling over 1.1 million characters. Data preprocessing involved removing special characters and extra whitespaces, then tokenizing the text into individual characters.

### Model Architectures
Two LSTM Char RNN models were established: a single-layer LSTM and a double-layer LSTM. Hyperparameters such as learning rate, number of iterations, embedding dimensions, hidden dimensions, and batch size were kept constant for both models.

### Data Augmentation and Regularization
Data augmentation techniques, including character swapping and random deletion, were applied to increase the variability of training data. A dropout rate of 0.5 was also implemented for regularization. These techniques were applied only to the superior single-layer LSTM model.

## Experiments and Results
### Training Setup
- All models were trained for 4000 iterations.
- Loss, perplexity, and accuracy were recorded at each interval.
- Model checkpoints were saved at every 200 iterations.

### Layer Number Comparison
The single-layer LSTM without data augmentation outperformed the double-layer LSTM. Key metrics:
- **Single-layer LSTM**: Loss = 1.615, Perplexity = 5.030, Accuracy = 0.535
- **Double-layer LSTM**: Loss = 1.789, Perplexity = 5.981, Accuracy = 0.475

### Data Augmentation Effect on Single-layer LSTM
Data augmentation and regularization slightly improved the performance of the single-layer LSTM. Key metrics:
- **Without Augmentation**: Loss = 1.615, Perplexity = 5.030, Accuracy = 0.535
- **With Augmentation**: Loss = 1.605, Perplexity = 4.977, Accuracy = 0.525

## Discussion
The single-layer LSTM was found to be more effective than the double-layer LSTM. Data augmentation and regularization contributed to overall stability in the training process, although the improvement in final performance was marginal.

## Conclusion
This research provides a comparison between single and double-layer LSTM architectures for Char RNN and demonstrates the impact of data augmentation and regularization techniques. Future work includes further hyperparameter optimization and validation through re-training.

## Acknowledgements
Special thanks to Sebastian Raschka for his video series on implementing a Character RNN in PyTorch, which greatly aided in the understanding and development of this project.

## References
- Karpathy, A. (2015). Tiny Shakespeare Dataset. [GitHub](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
- Raschka, S. (2021). Character RNN Example. [GitHub](https://github.com/rasbt/stat453-deep-learning-ss21/tree/main/L19/character-rnn)

## Contact
Wallace Elis Wayne Wefel - wewefel@ucsd.edu

## License
This project is licensed under the MIT License - see the LICENSE file for details.
