This project is a research/POC to understand Anomaly Detection benchmarks
 on our initial subset of logs.
 I have tried to understand the BETH cybersecurity dataset
 for anomaly detection and OoD analysis. The data was
 sourced from a novel honeypot tracking system recording
 both kernel-level process events and DNS network traffic. It
 contains real-world attacks in the presence of benign modern
 OS and cloud provider traffic, without the added complexity
 of noisy artificial user activity. This cleanliness is ideal
 for OoD analysis, such that each host in the dataset only
 contains one or two data-generating distributions. We also
 include baselines for anomaly detection trained on a subset
 of the BETH dataset: robust covariance, one-class SVM,
 iForest, and DoSE-SVM (with a VAE).

 Reference : https://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-033.pdf
 
