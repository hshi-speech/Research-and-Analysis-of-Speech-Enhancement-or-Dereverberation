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
|| ------ 7. Acknowledge  


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


#### 3.2.2 Inputs module  


#### 3.2.3 Phase module  


#### 3.2.4 Enhancement module  


#### 3.2.5 Post-processing module



### 3.2 Time domain speech enhancement (dereverberation)  



## 4. Public datasets  


## 5. Performance comparison  


## 6. Future trends  


## 7. Acknowledge  
Up to now, in the course of one and a half years of study, I would like to thank my tutors, Prof. Wang (Longbiao Wang. Tianjin University, China.) and Li (Sheng Li. National Institute of Information and Communications Technology (NICT), Japan.), Prof. Dang (Jianwu Dang. Tianjin University, China.), and the senior brother of the laboratory doctor, Meng Ge (Tianjin University, China) for their guidance and care for me. 
I hope I can successfully apply for a doctorate degree, and have the opportunity to discuss voice enhancement or other voice direction with you! 


