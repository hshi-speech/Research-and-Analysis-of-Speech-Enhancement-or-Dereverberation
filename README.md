# Research-and-Analysis-of-Speech-Enhancement-or-Dereverberation (RA-SED)
This repository contains some material of speech enhancement and dereverberation. On the one hand, I summarize this work for my further understanding. On the other hand, I hope that all beginners or masters interested in speech enhancement can ask me questions and make progress together.  
A lot of my summary is not very good, I hope you put forward corrections!  

><b>Advertisement：</b>  
>I would like to open source a speech enhancement toolkit in the near future, but there is currently no good way to do frame-level feature extraction. I would like to put the features in one file, but currently running on a small memory machine while reading and writing may run out of memory.  
>If you have a better way, please contact me!  
>Thank you!  
>`My email: hshi.cca@gmail.com, hshi_cca@tju.edu.cn (I will not be able to use this email after Jan. 2021!)`

## 0. Outlines
|| ------ 1. Overviews  
|| ------------ 1.1 What is speech enhancement (dereverberation)  
|| ------------ 1.2 Classification of speech enhancement (dereverberation)  
|| ------ 2. Traditional speech enhancement or dereverberation methods (I will show this part in the future.)  
|| ------ 3. Deep learning-based speech enhancement (dereverberation) methods  
|| ------------ 3.1 Basic framework  
|| ------------ 3.2 Frequency domain speech enhancement (dereverberation)  
|| ------------------ 3.2.1 Feature extraction module  
|| ------------------ 3.2.2 Inputs module  
|| ------------------ 3.2.3 Phase module  
|| ------------------ 3.2.4 Enhancement module  
|| ------------------ 3.2.5 Post-processing module  
|| ------------ 3.3 Time domain speech enhancement (dereverberation)  
|| ------ 4. Public datasets  
|| ------ 5. Performance comparison  
|| ------ 6. Future trends  
|| ------ 7. Speech signal processing group for worthy attention 
|| ------ 8. Acknowledge  


## 1. Overviews
We will give some basic introduction in this part. We first introduce what is speech enhancement (dereverberation) and its mathematical expression. Then we will give the classification of speech enhancement (dereverberation) which we summarized. 

### 1.1 What is speech enhancement (dereverberation)
In real life, microphone pickup, in addition to receiving voice, will also receive some noise and reverberation. Speech enhancement is aimed at noisy speech, want to get clean speech. But in fact, speech enhancement (dereverberation) will bring some distortion of noise signal, and can't restore clean speech. 
`Speech enhancement (dereverberation) is speech noise reduction (denoising).`  

The mathematical expression is as follows:  
$x = r * s + n$  
$s$ is speech signal (desired), $r$ is room impulse response (RIR)[[1]](https://www.researchgate.net/profile/Emanuel_Habets/publication/259991276_Room_Impulse_Response_Generator/links/5800ea5808ae1d2d72eae2a0/Room-Impulse-Response-Generator.pdf), $n$ is additive noise signal, and $x$ is microphone pickup signal, the noisy signal. 
The speech enhancement system wants to recover $s$ from $x$.  

Besides, some people think that it is necessary to remove the additive noise and reverberation[[2]](http://web.cse.ohio-state.edu/~wang.77/papers/HWWWMZ.taslp15.pdf) at the same time, but others think it is necessary to remove them separately[[3]](http://150.162.46.34:8080/icassp2017/pdfs/0005580.pdf). Therefore, there is no definite conclusion at present. But I prefer to remove additive noise and reverberation separately (`Personal opinion, for reference only`). 

<b>References:</b>  
[1] [E.A.P. Habets. Room impulse response generator[J]. Technische Universiteit Eindhoven, Tech. Rep, 2006, 2(2.4): 1.](https://www.researchgate.net/profile/Emanuel_Habets/publication/259991276_Room_Impulse_Response_Generator/links/5800ea5808ae1d2d72eae2a0/Room-Impulse-Response-Generator.pdf)  
[2] [K. Han, Y. Wang, D. Wang, et al. Learning spectral mapping for speech dereverberation and denoising[J]. IEEE/ACM TASLP, 2015, 23(6): 982-992.](http://web.cse.ohio-state.edu/~wang.77/papers/HWWWMZ.taslp15.pdf)  
[3] [Y. Zhao, Z. Wang, D. Wang. A two-stage algorithm for noisy and reverberant speech enhancement[C]//2017 IEEE ICASSP. IEEE, 2017: 5580-5584.](http://150.162.46.34:8080/icassp2017/pdfs/0005580.pdf)  

### 1.2 Classification of speech enhancement (dereverberation)  
In fact, I prefer to refer to those models that need to restore speech signal for human auditory perception as speech enhancement. Those for other tasks, such as automatic speech recognition (ASR) or speaker recognition, are called <b>feature enhancement</b>. Feature enhancement is to design the input and output of the model for a specific task, and does not need to be reconstructed to speech signal.  `So in this git, our speech enhancement (dereverberation) models are all for human auditory perception experiences.`  

According to methods, speech enhancement can be divided into traditional speech enhancement and deep learning (machine learning) based speech enhancement. 
Traditional speech enhancement methods need to rely on some assumptions. When dealing with the non-stationary features, the performance will degrade greatly, and some new distortion and noise will be introduced, such as music noise[[4]](https://d1wqtxts1xzle7.cloudfront.net/30803852/Yannis_Stylianou_Progress_in_Nonlinear_Speech_P.pdf?1362353818=&response-content-disposition=inline%3B+filename%3DSpectral_analysis_of_speech_signals_usin.pdf&Expires=1593847717&Signature=XMqllmmYJErenzsb7pqXiGPRTnLEv2yYJriPbclTqJ02mqEr5s9K~nTEz1tsFemydhL3be6vYfNFelkSfF4wge-7TwXCNO-oo1bVVxSDavXSIo52Qb~ZDI3BSeejq6PupXF5c9i7tzJ5vIubGYb9mk2k72MSPtqPATkiQqUFFNA9R9XvOuCRgABIS-gxQzQf~X4Jz~7yoN~4e7T0-ihEws0-h9qBnDjTPUh0afz2XKpSaekJMtH0Mo7OE8MEKaU8o8gudSSeG01PlvhQOzNDhsu1~GYNu4rkOM2qKTkQ12y3mZ1CER9mMCx-jQmY0XcQIEJADZhzaq4~QluGGtGS~w__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA#page=227). 
With the continuous improvement of the method and the increase of computing resources, deep learning (machine learning) based speech enhancement shows stronger performance. 
Combined with the above, I roughly classify speech enhancement：  
* Traditional speech enhancement  
* Deep learning (machine learning) based speech enhancement  
  * Frequency domain based speech enhancement  
  * Time domain based speech enhancement  
  
In frequency domain speech enhancement, Fourier transform (e.g., short-time fourier transform, STFT) is used to transform time domain signal into frequency domain representation. 
While for time domain speech enhancement, waveform signals are directly used. 
What features are used and how they are handled will be explained in the following introduction.  

<b>References:</b>  
[4] [A. Hussain, M. Chetouani, S. Squartini, et al. Nonlinear speech enhancement: An overview[M]//Progress in nonlinear speech processing. Springer, Berlin, Heidelberg, 2007: 217-248.](https://d1wqtxts1xzle7.cloudfront.net/30803852/Yannis_Stylianou_Progress_in_Nonlinear_Speech_P.pdf?1362353818=&response-content-disposition=inline%3B+filename%3DSpectral_analysis_of_speech_signals_usin.pdf&Expires=1593847717&Signature=XMqllmmYJErenzsb7pqXiGPRTnLEv2yYJriPbclTqJ02mqEr5s9K~nTEz1tsFemydhL3be6vYfNFelkSfF4wge-7TwXCNO-oo1bVVxSDavXSIo52Qb~ZDI3BSeejq6PupXF5c9i7tzJ5vIubGYb9mk2k72MSPtqPATkiQqUFFNA9R9XvOuCRgABIS-gxQzQf~X4Jz~7yoN~4e7T0-ihEws0-h9qBnDjTPUh0afz2XKpSaekJMtH0Mo7OE8MEKaU8o8gudSSeG01PlvhQOzNDhsu1~GYNu4rkOM2qKTkQ12y3mZ1CER9mMCx-jQmY0XcQIEJADZhzaq4~QluGGtGS~w__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA#page=227)  


## 2. Traditional speech enhancement or dereverberation methods (I will show this part in the future.)


## 3. Deep learning-based speech enhancement (dereverberation) methods
### 3.1 Basic framework  
In section 1.2, I give the classification of speech enhancement, and their basic framework is as follows:  
  
![](https://github.com/hshi-cca/Research-and-Analysis-of-Speech-Enhancement-or-Dereverberation/blob/master/saving/github-pic.png)  
  
`In fact, each module in the diagram has no clear name. We are here to introduce the work more conveniently, so we give each part a name. (don't spray if you don't like it)` 
According to my experience, speech enhancement in frequency domain is divided into five modules, while speech enhancement in time domain is divided into three modules. 
That is because I feel that the current speech enhancement, we are generally improve the performance from these parts. 
Everyone's classification is different. Here I will sort it out according to my ideas. In many papers, the improved model may have more than one module. Here I only focus on the most important improvement in these papers to classify. 
I have been focusing on speech enhancement for about a year and a half. Of course, there are some models that I have not paid attention to. Please put forward your correction by email.  

### 3.2 Frequency domain speech enhancement (dereverberation)  
The speech enhancement algorithm in frequency domain is more perfect than that in time domain. 
I divide it into five modules. 
I will show you how to improve the frequency domain speech enhancement algorithm from these five parts.  

#### 3.2.1 Feature extraction module  
<b>Input Feature</b>  
The input feature is very important to the learning of neural network. 
* <b>Mel frequency power spectrum (MFP)</b> was used for speech enhancement in INTERSPEECH 2013 [[5]](https://bio-asplab.citi.sinica.edu.tw/paper/conference/lu2013speech.pdf). 
* At present, the most common feature is <b>the magnitude of spectrogram</b>. 
* <b>Log processing of the magnitude of spectrogram</b>[[6]](http://staff.ustc.edu.cn/~jundu/Publications/publications/SPL2014_Xu.pdf) is more suitable for human hearing. 
* Moreover, in the way of multi-target learning (MTL) and combining various features, e.g., <b>mel-frequency cepstral coefficients (MFCC)</b>, as input and output, the network can also achieve good results[[7]](https://arxiv.org/pdf/1703.07172.pdf).  

<b>Learning Targets</b>  
* Using the nonlinear mapping ability of neural network, we can map the spectrum directly, which is called <b>mapping</b> approach. 
* Masking approach is another common learning targets for speech enhancement. 
 * <b>Ideal binary mask (IBM)</b>[[9]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4293540/) and 
 * <b>ideal ratio mask (IRM)</b>[[9]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4293540/) are common masking approaches based on the computational auditory scene analysis (CASA)[[8]](https://ieeexplore.ieee.org/document/4429320?denied=) theory.  

 * Besides, considering the influence of phase to speech enhancement, the <b>complex ideal ratio mask (cIRM)</b>[[10]](http://homes.sice.indiana.edu/williads/publication_files/williamsonetal.cRM.ICASSP2016.pdf) and 
 * the <b>phase-sensitive spectrum approximation (PSA)</b>[[11]](https://merl.com/publications/docs/TR2015-031.pdf) are also proposed and used.  

<b>Other Feature Processing</b>  
* In addition to Fourier transform (FT), 
* other transform methods, such as Z-transform, will also be used in feature extraction of speech enhancement. 
* Different filter banks have great influence on feature extraction. 
* Whether to normalize the features, and how to normalize the features, especially for those MTL models, I will sort them out and improve them in the future work!  

<b>References:</b>  
[5] [X. Lu, Y. Tsao, S. Matsuda, et al. Speech enhancement based on deep denoising autoencoder[C]//Interspeech. 2013, 2013: 436-440.](https://bio-asplab.citi.sinica.edu.tw/paper/conference/lu2013speech.pdf)  
[6] [Y. Xu, J. Du, L. Dai, et al. An experimental study on speech enhancement based on deep neural networks[J]. IEEE Signal processing letters, 2013, 21(1): 65-68.](http://staff.ustc.edu.cn/~jundu/Publications/publications/SPL2014_Xu.pdf)  
[7] [Y. Xu, J. Du, Z. Huang, et al. Multi-objective learning and mask-based post-processing for deep neural network based speech enhancement[J]. arXiv preprint arXiv:1703.07172, 2017.](https://arxiv.org/pdf/1703.07172.pdf)  
[8] [D. Wang, G. J. Brown. Computational auditory scene analysis: Principles, algorithms, and applications[M]. Wiley-IEEE press, 2006.](https://ieeexplore.ieee.org/document/4429320?denied=)  
[9] [Y. Wang, A. Narayanan, D. Wang. On training targets for supervised speech separation[J]. IEEE/ACM TASLP, 2014, 22(12): 1849-1858.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4293540/)  
[10] [D. S. Williamson, Y. Wang, D. Wang. Complex ratio masking for joint enhancement of magnitude and phase[C]//2016 IEEE ICASSP. IEEE, 2016: 5220-5224.](http://homes.sice.indiana.edu/williads/publication_files/williamsonetal.cRM.ICASSP2016.pdf)  
[11] [H. Erdogan, J. R. Hershey, S. Watanabe, et al. Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks[C]//2015 IEEE ICASSP. IEEE, 2015: 708-712.
](https://merl.com/publications/docs/TR2015-031.pdf)  

#### 3.2.2 Inputs module  
In fact, this part should be combined with 3.2.1 to say what features need to be input to improve the performance of our speech enhancement system. 
* [[7]](https://arxiv.org/pdf/1703.07172.pdf) shows that some complimentary features can improve the enhancement performance. 
* In recently years, adding some symbol information [[12]](https://arxiv.org/pdf/1904.13142.pdf) and 
* text information[[13]](http://www.kecl.ntt.co.jp/icl/signal/kinoshita/publications/Interspeech15/IS150674.pdf) to the network can improve the performance of speech enhancement.  
* Besides, it is also necessary to select whether frame-level[[6]](http://staff.ustc.edu.cn/~jundu/Publications/publications/SPL2014_Xu.pdf) features or 
* utterance-level features[[18]](https://www.researchgate.net/profile/Ke_Tan6/publication/325542192_A_Convolutional_Recurrent_Neural_Network_for_Real-Time_Speech_Enhancement/links/5b91955292851c78c4f3d317/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement.pdf) are needed, and 
* whether or not frame expansion[[6]](http://staff.ustc.edu.cn/~jundu/Publications/publications/SPL2014_Xu.pdf) is needed and how many frames are spliced. 
* Moreover, there are other information that can help improve the performance of speech enhancement, and I will continue to add it.  

<b>References:</b>  
[12] [C. Liao, Y. Tsao, X. Lu, et al. Incorporating symbolic sequential modeling for speech enhancement[J]. arXiv preprint arXiv:1904.13142, 2019.](https://arxiv.org/pdf/1904.13142.pdf)  
[13] [K. Kinoshita, M. Delcroix, A. Ogawa, et al. Text-informed speech enhancement with deep neural networks[C]//Sixteenth Annual Conference of the International Speech Communication Association. 2015.](http://www.kecl.ntt.co.jp/icl/signal/kinoshita/publications/Interspeech15/IS150674.pdf)  
[18] [K. Tan, D. Wang. A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement[C]//Interspeech. 2018: 3229-3233.](https://www.researchgate.net/profile/Ke_Tan6/publication/325542192_A_Convolutional_Recurrent_Neural_Network_for_Real-Time_Speech_Enhancement/links/5b91955292851c78c4f3d317/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement.pdf)  

#### 3.2.3 Phase module  
After a long time of research, researchers found that phase information is also important[[14]](https://maxwell.ict.griffith.edu.au/spl/publications/papers/spcom11_phase_enhance.pdf) for speech enhancement. 
However, due to the fact that there is no fixed law of phase information and there are some problems such as phase winding, it is difficult to deal with phase information.  Therefore, in the early speech enhancement based on deep learning, only magnitude information is processed, and then the waveform is reconstructed with noisy phase.  

<b>Inconsistency of ISTFT</b>  
* Besides, considering the inconsistency between the enhanced spectrogram and the noisy phase when inverse STFT (ISTFT)[[14]](https://arxiv.org/pdf/1811.08521.pdf).  

<b>Iterative Signal Rconstruction</b>  
* The simplest way to deal with phase is by iterating[[16]](http://web.cse.ohio-state.edu/~wang.77/papers/HWWWMZ.taslp15.pdf). Firstly, the enhanced amplitude information and noisy phase are used to reconstruct the waveform, and then the phase of the reconstructed waveform is extracted, and then the re-extracted phase is used to reconstruct the waveform. Through repeated iterations, the effect can be improved.  

<b>A Way of Bypassing or Utilizing Phase Information</b>  
* PSA[[11]](https://merl.com/publications/docs/TR2015-031.pdf) can make use of the phase information and improve to a certain extent. 
* cIRM[[10]](http://homes.sice.indiana.edu/williads/publication_files/williamsonetal.cRM.ICASSP2016.pdf) and 
* time domain speech enhancement can bypass the phase problem.  

<b>Phase reconstruction</b>  
* Moreover, phase information can also be used or even predicted through the network, combined with magnitude information[[17]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YinD.3057.pdf).  

<b>References:</b>   
[14] [S. Wisdom, J. R. Hershey, K. Wilson, et al. Differentiable consistency constraints for improved deep speech enhancement[C]//ICASSP 2019-2019 IEEE ICASSP. IEEE, 2019: 900-904.](https://arxiv.org/pdf/1811.08521.pdf)  
[15] [K. Paliwal, K. Wójcicki, B. Shannon. The importance of phase in speech enhancement[J]. speech communication, 2011, 53(4): 465-494.](https://maxwell.ict.griffith.edu.au/spl/publications/papers/spcom11_phase_enhance.pdf)   
[16] [K. Han, Y. Wang, D. Wang, et al. Learning spectral mapping for speech dereverberation and denoising[J]. IEEE/ACM TASLP, 2015, 23(6): 982-992.](http://web.cse.ohio-state.edu/~wang.77/papers/HWWWMZ.taslp15.pdf)  
[17] [D. Yin, C. Luo, Z. Xiong, et al. PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network[C]//AAAI. 2020: 9458-9465.](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YinD.3057.pdf)  

#### 3.2.4 Enhancement module  
<b>Improvement of Neural Network</b>  
The choice of model is very important. 
* From the beginning of DNN[[6]](http://staff.ustc.edu.cn/~jundu/Publications/publications/SPL2014_Xu.pdf), 
* to CNN[[19]](https://arxiv.org/pdf/1609.07132.pdf) 
* and RNN[[20]](http://home.ustc.edu.cn/~sunlei17/pdf/MULTIPLE-TARGET.pdf), the model is gradually powerful, and the performance of speech enhancement is also improving. 
* The combination of different networks also has a good effect[[21]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1477.pdf). 

<b>Design of Network Structure</b>  
* U-NET structure[[22]](https://openreview.net/pdf?id=SkeRTsAcYm) has been proved to be effective in many tasks. 
* By using the idea of generation and confrontation, generative adversarial network (GAN)[[23]](https://ieeexplore.ieee.org/abstract/document/8462068) uses the discriminator to judge the effect of the generator, and by introducing some information of the discriminator, to a certain extent, it can alleviate the defect that mean squared error (MSE) as a loss function may not be suitable for human hearing. 

<b>Loss Function</b>  
* In order to be more suitable for human hearing, some speech enhancement systems also use evaluation index as loss function, e.g., short-time objective intelligibility (STOI)[[24]](https://arxiv.org/pdf/1802.00604.pdf), [[25]](https://cliffzhao.github.io/Publications/ZXGZ.icassp18.pdf). 
* Noise (domain) information can be used in the model in the form of loss[[36]](https://arxiv.org/pdf/1807.07501.pdf).  

<b>Attention Mechanism</b>
* The attention mechanism[[26]](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) is becoming more and more common, and the attention-based approach
has better generalization ability to unseen noise conditions[[27]](http://www.npu-aslp.org/lxie/papers/2019ICASSP-XiangHao.pdf). 
* Besides, different researcheres use the attention mechanism to model different units, e.g., [[28]](https://arxiv.org/pdf/2001.11542.pdf) use attention mechanism to model different signal channels.  

<b>Training Strategy</b>  
Considering some characteristics of the network, training or designing some structures can improve the enhanced performance. 
* Transfer learning[[29]](http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/pdfs/Yong_ISCSLP2014.pdf) can improve the performance of the model on small databases by training on large databases and fine tuning on small databases. 
* In the way of multi-target learning (MTL) and combining various features, e.g., mel-frequency cepstral coefficients (MFCC), as input and output, the network can also achieve good results[[7]](https://arxiv.org/pdf/1703.07172.pdf). 
* Different networks are trained according to different SNR[[30]](https://www.researchgate.net/profile/Yu_Tsao/publication/307889660_SNR-Aware_Convolutional_Neural_Network_Modeling_for_Speech_Enhancement/links/57ee69e908ae280dd0ad5866.pdf). 
* [[31]](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005054.pdf) uses progressive learning to enhance the results step by step. 
The stacking of networks[[32]](https://web.cse.ohio-state.edu/~wang.77/papers/Wang-Wang1.icassp17.pdf) also shows the effect.  

<b>Fusion Strategy</b>  
* Moerover, masking and mapping approaches show different effects on different scenarios, which shows some complementary. [[33]](https://ieeexplore.ieee.org/abstract/document/9054661/) proposes minimum difference masks to utilize this complementary, and fuses the spectrograms. 
* [[34]](https://arxiv.org/pdf/1812.08914.pdf) fuses the enhanced signal in time domain and enhanced signal in frequency domain, which also showed the effect.  

<b>Mutual Enhancement</b>  
The combination of enhancement task and other tasks can enhance each other. 
* The phonetic posteriorgrams (PPG) shows a certain correlation with enhancement[[35]](https://ieeexplore.ieee.org/abstract/document/9054334). 

<b>References:</b>   
[19] [S. R. Park, J. Lee. A fully convolutional neural network for speech enhancement[J]. arXiv preprint arXiv:1609.07132, 2016.](https://arxiv.org/pdf/1609.07132.pdf)  
[20] [L. Sun, J. Du, L. Dai, et al. Multiple-target deep learning for LSTM-RNN based speech enhancement[C]//2017 HSCMA. IEEE, 2017: 136-140.](http://home.ustc.edu.cn/~sunlei17/pdf/MULTIPLE-TARGET.pdf)  
[21] [M. Ge, L. Wang, N. Li, et al. Environment-Dependent Attention-Driven Recurrent Convolutional Neural Network for Robust Speech Enhancement[C]//INTERSPEECH. 2019: 3153-3157.](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1477.pdf)  
[22] [H. Choi, J. Kim, J. Huh, et al. Phase-aware speech enhancement with deep complex u-net[C]//International Conference on Learning Representations. 2018.](https://openreview.net/pdf?id=SkeRTsAcYm)  
[23] [M. H. Soni, N. Shah, H. A. Patil. Time-frequency masking-based speech enhancement using generative adversarial network[C]//2018 IEEE ICASSP. IEEE, 2018: 5039-5043.](https://ieeexplore.ieee.org/abstract/document/8462068)  
[24] [M. Kolbæk, Z. Tan, J. Jensen. Monaural speech enhancement using deep neural networks by maximizing a short-time objective intelligibility measure[C]//2018 IEEE ICASSP. IEEE, 2018: 5059-5063.](https://arxiv.org/pdf/1802.00604.pdf)  
[25] [Y. Zhao, B. Xu, R. Giri, et al. Perceptually guided speech enhancement using deep neural networks[C]//2018 IEEE ICASSP. IEEE, 2018: 5074-5078.](https://cliffzhao.github.io/Publications/ZXGZ.icassp18.pdf)  
[26] [A. Vaswani, N. Shazeer, N. Parmar, et al. Attention is all you need[C]//Advances in neural information processing systems. 2017: 5998-6008.](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)  
[27] [X. Hao, C. Shan, Y. Xu, et al. An attention-based neural network approach for single channel speech enhancement[C]// 2019 IEEE ICASSP. IEEE, 2019: 6895-6899.](http://www.npu-aslp.org/lxie/papers/2019ICASSP-XiangHao.pdf)  
[28] [B. Tolooshams, R. Giri, A. H. Song, et al. Channel-Attention Dense U-Net for Multichannel Speech Enhancement[C]//ICASSP 2020 ICASSP. IEEE, 2020: 836-840.](https://arxiv.org/pdf/2001.11542.pdf)  
[29] [Y. Xu, J. Du, L. Dai, et al. Cross-language transfer learning for deep neural network based speech enhancement[C]//The 9th International Symposium on Chinese Spoken Language Processing. IEEE, 2014: 336-340.](http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/pdfs/Yong_ISCSLP2014.pdf)  
[30] [S. Fu, Y. Tsao, X. Lu. SNR-Aware Convolutional Neural Network Modeling for Speech Enhancement[C]//Interspeech. 2016: 3768-3772.](https://www.researchgate.net/profile/Yu_Tsao/publication/307889660_SNR-Aware_Convolutional_Neural_Network_Modeling_for_Speech_Enhancement/links/57ee69e908ae280dd0ad5866.pdf)  
[31] [T. Gao, J. Du, L. Dai, et al. Densely connected progressive learning for lstm-based speech enhancement[C]//2018 IEEE ICASSP. IEEE, 2018: 5054-5058.](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005054.pdf)  
[32] [Z. Wang, D. Wang. Recurrent deep stacking networks for supervised speech separation[C]//2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017: 71-75.](https://web.cse.ohio-state.edu/~wang.77/papers/Wang-Wang1.icassp17.pdf)  
[33] [H. Shi, L. Wang, M. Ge, et al. Spectrograms Fusion with Minimum Difference Masks Estimation for Monaural Speech Dereverberation[C]//ICASSP ICASSP. IEEE, 2020: 7544-7548.](https://ieeexplore.ieee.org/abstract/document/9054661/)  
[34] [J. Kim, J. Yoo, S. Chun, et al. Multi-domain processing via hybrid denoising networks for speech enhancement[J]. arXiv preprint arXiv:1812.08914, 2018.](https://arxiv.org/pdf/1812.08914.pdf)  
[35] [Z. Du, M. Lei, J. Han, et al. Pan: Phoneme-Aware Network for Monaural Speech Enhancement[C]//ICASSP 2020 IEEE ICASSP. IEEE, 2020: 6634-6638.](https://ieeexplore.ieee.org/abstract/document/9054334)  
[36] [C. Liao, Y. Tsao, H. Lee, et al. Noise adaptive speech enhancement using domain adversarial training[J]. arXiv preprint arXiv:1807.07501, 2018.](https://arxiv.org/pdf/1807.07501.pdf)


#### 3.2.5 Post-processing module


<b>References:</b>  

### 3.2 Time domain speech enhancement (dereverberation)  
Time domain speech enhancement can enhance the time domain speech waveform signal in the form of end-to-end because it does not need to consider the characteristics and can bypass the phase problem. Moreover, in many literatures, I found that the time domain speech enhancement basically uses convolutional neural network (CNN) or fully convolutional neural network (FCN) as the network structure. 

#### 3.2.1 Why use FCN (CNN) in time-domain


#### 3.2.2 Neural network structure


#### 3.2.3 Loss function


<b>References:</b>  

## 4. Public datasets  




## 5. Performance comparison  


<b>References:</b>  

## 6. Future trends  


## 8. Acknowledge  
Up to now, in the course of one and a half years of study, I would like to thank my tutors, Prof. Wang (Longbiao Wang. Tianjin University, China.) and Li (Sheng Li. National Institute of Information and Communications Technology (NICT), Japan.), Prof. Dang (Jianwu Dang. Tianjin University, China.), and the senior brother of the laboratory doctor, Meng Ge (Tianjin University, China) for their guidance and care for me. 
I hope I can successfully apply for a doctorate degree, and have the opportunity to discuss voice enhancement or other voice direction with you! 


