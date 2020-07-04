# Research-and-Analysis-of-Speech-Enhancement-or-Dereverberation (RA-SED)
This repository contains some material of speech enhancement and dereverberation. On the one hand, I summarize this work for my further understanding. On the other hand, I hope that all beginners or masters interested in speech enhancement can ask me questions and make progress together.  
A lot of my summary is not very good, I hope you put forward corrections!  

<b>Advertisement：</b>  
I would like to open source a speech enhancement toolkit in the near future, but there is currently no good way to do frame-level feature extraction. I would like to put the features in one file, but currently running on a small memory machine while reading and writing may run out of memory.  
If you have a better way, please contact me!  
Thank you!  

`
My email: hshi.cca@gmail.com, hshi_cca@tju.edu.cn (I will not be able to use this email after Jan. 2021!)
`

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
The input feature is very important to the learning of neural network. 
Mel frequency power spectrum (MFP) was used for speech enhancement in INTERSPEECH 2013 [[5]](https://bio-asplab.citi.sinica.edu.tw/paper/conference/lu2013speech.pdf). 
At present, the most common feature is the magnitude of spectrogram. Log processing of the magnitude of spectrogram[[6]](http://staff.ustc.edu.cn/~jundu/Publications/publications/SPL2014_Xu.pdf) is more suitable for human hearing. 
Using the nonlinear mapping ability of neural network, we can map the spectrum directly, which is called mapping approach. 
Masking approach is another common learning targets for speech enhancement. 
Ideal binary mask (IBM)[[9]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4293540/) and ideal ratio mask (IRM)[[9]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4293540/) are common masking approaches based on the computational auditory scene analysis (CASA)[[8]](https://ieeexplore.ieee.org/document/4429320?denied=) theory. 

Moreover, in the way of multi-target learning (MTL) and combining various features, e.g., mel-frequency cepstral coefficients (MFCC), as input and output, the network can also achieve good results[[7]](https://arxiv.org/pdf/1703.07172.pdf).  

Besides, considering the influence of phase to speech enhancement, the complex ideal ratio mask (cIRM)[[10]](http://homes.sice.indiana.edu/williads/publication_files/williamsonetal.cRM.ICASSP2016.pdf) and the phase-sensitive spectrum approximation (PSA)[[11]](https://merl.com/publications/docs/TR2015-031.pdf) are also proposed and used.  

In addition to Fourier transform, other transform methods, such as Z-transform, will also be used in feature extraction of speech enhancement. 
Different filter banks have great influence on feature extraction.  

Whether to normalize the features, and how to normalize the features, especially for those MTL models, I will sort them out and improve them in the future work!  

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
[[7]](https://arxiv.org/pdf/1703.07172.pdf) shows that some complimentary features can improve the enhancement performance. 
In recently years, adding some symbol information [[12]](https://arxiv.org/pdf/1904.13142.pdf) and text information[[13]](http://www.kecl.ntt.co.jp/icl/signal/kinoshita/publications/Interspeech15/IS150674.pdf) to the network can improve the performance of speech enhancement.  

Besides, it is also necessary to select whether frame-level[[6]](http://staff.ustc.edu.cn/~jundu/Publications/publications/SPL2014_Xu.pdf) features or utterance-level features[[18]](https://www.researchgate.net/profile/Ke_Tan6/publication/325542192_A_Convolutional_Recurrent_Neural_Network_for_Real-Time_Speech_Enhancement/links/5b91955292851c78c4f3d317/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement.pdf) are needed, and whether or not frame expansion[[6]](http://staff.ustc.edu.cn/~jundu/Publications/publications/SPL2014_Xu.pdf) is needed and how many frames are spliced.

Moreover, there are other information that can help improve the performance of speech enhancement, and I will continue to add it.  

<b>References:</b>  
[12] [C. Liao, Y. Tsao, X. Lu, et al. Incorporating symbolic sequential modeling for speech enhancement[J]. arXiv preprint arXiv:1904.13142, 2019.](https://arxiv.org/pdf/1904.13142.pdf)  
[13] [K. Kinoshita, M. Delcroix, A. Ogawa, et al. Text-informed speech enhancement with deep neural networks[C]//Sixteenth Annual Conference of the International Speech Communication Association. 2015.](http://www.kecl.ntt.co.jp/icl/signal/kinoshita/publications/Interspeech15/IS150674.pdf)  
[18] [K. Tan, D. Wang. A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement[C]//Interspeech. 2018: 3229-3233.](https://www.researchgate.net/profile/Ke_Tan6/publication/325542192_A_Convolutional_Recurrent_Neural_Network_for_Real-Time_Speech_Enhancement/links/5b91955292851c78c4f3d317/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement.pdf)  

#### 3.2.3 Phase module  
After a long time of research, researchers found that phase information is also important[[14]](https://maxwell.ict.griffith.edu.au/spl/publications/papers/spcom11_phase_enhance.pdf) for speech enhancement. 
However, due to the fact that there is no fixed law of phase information and there are some problems such as phase winding, it is difficult to deal with phase information.  Therefore, in the early speech enhancement based on deep learning, only magnitude information is processed, and then the waveform is reconstructed with noisy phase. 
Besides, considering the inconsistency between the enhanced spectrogram and the noisy phase when inverse STFT (ISTFT)[[14]](https://arxiv.org/pdf/1811.08521.pdf). 
The simplest way to deal with phase is by iterating[[16]](http://web.cse.ohio-state.edu/~wang.77/papers/HWWWMZ.taslp15.pdf). Firstly, the enhanced amplitude information and noisy phase are used to reconstruct the waveform, and then the phase of the reconstructed waveform is extracted, and then the re extracted phase is used to reconstruct the waveform. Through repeated iterations, the effect can be improved.  

PSA[[11]](https://merl.com/publications/docs/TR2015-031.pdf) can make use of the phase information and improve to a certain extent. 
cIRM[[10]](http://homes.sice.indiana.edu/williads/publication_files/williamsonetal.cRM.ICASSP2016.pdf) and time domain speech enhancement can bypass the phase problem.  

Moreover, phase information can also be used or even predicted through the network, combined with magnitude information[[17]](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YinD.3057.pdf).  

<b>References:</b>   
[14] [S. Wisdom, J. R. Hershey, K. Wilson, et al. Differentiable consistency constraints for improved deep speech enhancement[C]//ICASSP 2019-2019 IEEE ICASSP. IEEE, 2019: 900-904.](https://arxiv.org/pdf/1811.08521.pdf)  
[15] [K. Paliwal, K. Wójcicki, B. Shannon. The importance of phase in speech enhancement[J]. speech communication, 2011, 53(4): 465-494.](https://maxwell.ict.griffith.edu.au/spl/publications/papers/spcom11_phase_enhance.pdf)   
[16] [K. Han, Y. Wang, D. Wang, et al. Learning spectral mapping for speech dereverberation and denoising[J]. IEEE/ACM TASLP, 2015, 23(6): 982-992.](http://web.cse.ohio-state.edu/~wang.77/papers/HWWWMZ.taslp15.pdf)  
[17] [D. Yin, C. Luo, Z. Xiong, et al. PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network[C]//AAAI. 2020: 9458-9465.](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-YinD.3057.pdf)  

#### 3.2.4 Enhancement module  



<b>References:</b>   

#### 3.2.5 Post-processing module


<b>References:</b>  

### 3.2 Time domain speech enhancement (dereverberation)  



<b>References:</b>  

## 4. Public datasets  




## 5. Performance comparison  


<b>References:</b>  

## 6. Future trends  


## 8. Acknowledge  
Up to now, in the course of one and a half years of study, I would like to thank my tutors, Prof. Wang (Longbiao Wang. Tianjin University, China.) and Li (Sheng Li. National Institute of Information and Communications Technology (NICT), Japan.), Prof. Dang (Jianwu Dang. Tianjin University, China.), and the senior brother of the laboratory doctor, Meng Ge (Tianjin University, China) for their guidance and care for me. 
I hope I can successfully apply for a doctorate degree, and have the opportunity to discuss voice enhancement or other voice direction with you! 


