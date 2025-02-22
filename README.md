# s15erav3

s15erav3 is custom DeepSeek v3 model incorporating **Multi-Head Latent Attention (MLHA)** and **Mixture of Experts (MoE)** with **Loss-less Load Balancing** for optimized training efficiency.

## 🚀 Features

- **Multi-Head Latent Attention (MLHA)**: Enhances attention efficiency by selectively focusing on latent representations.
- **Mixture of Experts (MoE)**: Dynamically routes inputs to different expert layers for optimized computation.
- **Loss-less Load Balancing**: Ensures even distribution of token routing, preventing expert underutilization.
- **Flexible Configuration**: Adjustable hyperparameters via `config.yaml`.
- **Checkpointing Support**: Saves and resumes training from `checkpoints/`.

### Configuration
- **hidden_size**: 512
- **intermediate_size**: 768
- **sequence_length**: 512
- **num_experts**: 8
- **num_shared_experts**: 1
- **top_k_experts**: 2
## Training logs:
- Trained on _input.txt_ dataset of IPL commentary

```
loaded 1019673 tokens
1 epoch = 248 batches
step0 | loss: 11.248027801513672 | dt: 2584.80ms | tok/sec:  1584.65
Checkpoint saved at checkpoints/checkpoint_0.pth
Validation: (Step 0), Generated text: Kohli comes forward and  deathscuttingPres surprise starvedurb �influ imported enslavement gently Viralalfa septaster cause protr inst anchored sac feasts recipient pediatric lance tomorrowru Nebctors sworn Malays County cardboard pedestrianvementbleurtles Leaddrivenoffic
Forwardatives astronomical fluctureligious autom directive analogue remote cart
step10 | loss: 5.991911888122559 | dt: 1914.19ms | tok/sec:  2139.81
step20 | loss: 5.43084716796875 | dt: 2018.36ms | tok/sec:  2029.37
step30 | loss: 4.980795860290527 | dt: 1993.93ms | tok/sec:  2054.23
step40 | loss: 4.496359825134277 | dt: 1997.39ms | tok/sec:  2050.68
step50 | loss: 4.317827224731445 | dt: 1975.43ms | tok/sec:  2073.48
step60 | loss: 3.785248041152954 | dt: 1971.84ms | tok/sec:  2077.25
step70 | loss: 4.258542537689209 | dt: 1981.73ms | tok/sec:  2066.88
step80 | loss: 3.772378921508789 | dt: 1986.25ms | tok/sec:  2062.18
step90 | loss: 3.6067075729370117 | dt: 2027.12ms | tok/sec:  2020.60
step100 | loss: 3.6658847332000732 | dt: 2028.67ms | tok/sec:  2019.06
step110 | loss: 3.5033512115478516 | dt: 1977.57ms | tok/sec:  2071.23
step120 | loss: 3.3820478916168213 | dt: 1983.34ms | tok/sec:  2065.21
step130 | loss: 3.355147123336792 | dt: 1989.33ms | tok/sec:  2058.99
...
step950 | loss: 2.0374045372009277 | dt: 2002.67ms | tok/sec:  2045.27
step960 | loss: 2.0979998111724854 | dt: 2008.70ms | tok/sec:  2039.13
step970 | loss: 2.2532026767730713 | dt: 2008.14ms | tok/sec:  2039.70
step980 | loss: 1.9508591890335083 | dt: 1995.59ms | tok/sec:  2052.53
step990 | loss: 1.912624478340149 | dt: 2011.06ms | tok/sec:  2036.74
step1000 | loss: 2.127336263656616 | dt: 2012.30ms | tok/sec:  2035.48
Validation: (Step 1000), Generated text: Kohli comes forward and 16) to the bat on the bat on the bat on the bat on the bat on the bat on the bat on the bat on the bat on the bat on the bat on the bat on the bat on the bat on the bat on the
step1010 | loss: 1.9026015996932983 | dt: 1996.67ms | tok/sec:  2051.42
step1020 | loss: 1.897756576538086 | dt: 1997.55ms | tok/sec:  2050.51
step1030 | loss: 1.8216891288757324 | dt: 1995.97ms | tok/sec:  2052.13
step1040 | loss: 2.288748264312744 | dt: 2008.67ms | tok/sec:  2039.16
...
step2420 | loss: 0.7007253170013428 | dt: 2011.25ms | tok/sec:  2036.55
step2430 | loss: 0.6297969222068787 | dt: 1987.94ms | tok/sec:  2060.43
step2440 | loss: 0.6408869624137878 | dt: 1984.66ms | tok/sec:  2063.83
step2450 | loss: 0.7099995017051697 | dt: 2005.53ms | tok/sec:  2042.36
step2460 | loss: 0.6584897637367249 | dt: 1999.56ms | tok/sec:  2048.45
step2470 | loss: 0.705953061580658 | dt: 1995.62ms | tok/sec:  2052.50
step2480 | loss: 0.7181088328361511 | dt: 2043.42ms | tok/sec:  2004.49
step2490 | loss: 0.647209882736206 | dt: 1991.23ms | tok/sec:  2057.02
step2500 | loss: 0.6999458074569702 | dt: 2017.83ms | tok/sec:  2029.90
Validation: (Step 2500), Generated text: Kohli comes forward and 19 overs
19 overs
19 overs
19 overs
19 overs
19 overs
19 overs
19 overs
19 overs
19 overs
19 overs
19 overs
19
step2510 | loss: 0.6633559465408325 | dt: 2053.89ms | tok/sec:  1994.26
...
step4520 | loss: 0.23203018307685852 | dt: 2026.33ms | tok/sec:  2021.38
step4530 | loss: 0.20117437839508057 | dt: 2021.01ms | tok/sec:  2026.71
step4540 | loss: 0.24878333508968353 | dt: 1990.79ms | tok/sec:  2057.48
step4550 | loss: 0.23195703327655792 | dt: 1995.33ms | tok/sec:  2052.79
step4560 | loss: 0.23288914561271667 | dt: 2016.28ms | tok/sec:  2031.46
step4570 | loss: 0.2281227558851242 | dt: 2005.32ms | tok/sec:  2042.57
step4580 | loss: 0.23429851233959198 | dt: 2034.86ms | tok/sec:  2012.91
step4590 | loss: 0.22011837363243103 | dt: 2009.88ms | tok/sec:  2037.93
step4600 | loss: 0.2415919303894043 | dt: 2008.58ms | tok/sec:  2039.25
step4610 | loss: 0.2089819312095642 | dt: 2022.69ms | tok/sec:  2025.03
step4620 | loss: 0.18834692239761353 | dt: 2011.13ms | tok/sec:  2036.67
step4630 | loss: 0.22154851257801056 | dt: 1998.65ms | tok/sec:  2049.39
step4640 | loss: 0.21229247748851776 | dt: 2021.81ms | tok/sec:  2025.91
step4650 | loss: 0.19291694462299347 | dt: 1990.58ms | tok/sec:  2057.70
step4660 | loss: 0.21274183690547943 | dt: 2020.97ms | tok/sec:  2026.75
step4670 | loss: 0.1889066994190216 | dt: 2000.44ms | tok/sec:  2047.55
...
step5480 | loss: 0.19068343937397003 | dt: 2012.34ms | tok/sec:  2035.44
step5490 | loss: 0.18867623805999756 | dt: 1997.02ms | tok/sec:  2051.05
step5500 | loss: 0.1550283581018448 | dt: 2005.25ms | tok/sec:  2042.64
Validation: (Step 5500), Generated text: Kohli comes forward and 1st Inns
B 1st Inns
B 1st Inns
B 1st Inns
B 1st Inns
B 1st Inns
B 1st Inns
B 1
step5510 | loss: 0.18661533296108246 | dt: 1995.89ms | tok/sec:  2052.22
step5520 | loss: 0.16412048041820526 | dt: 1999.54ms | tok/sec:  2048.47
step5530 | loss: 0.15810267627239227 | dt: 2010.64ms | tok/sec:  2037.16
step5540 | loss: 0.18578635156154633 | dt: 2002.09ms | tok/sec:  2045.86
step5550 | loss: 0.18032599985599518 | dt: 1999.14ms | tok/sec:  2048.89
...
step6810 | loss: 0.11860775202512741 | dt: 2062.42ms | tok/sec:  1986.01
step6820 | loss: 0.11925489455461502 | dt: 2013.88ms | tok/sec:  2033.89
step6830 | loss: 0.12650495767593384 | dt: 2000.03ms | tok/sec:  2047.97
step6840 | loss: 0.14004625380039215 | dt: 2017.64ms | tok/sec:  2030.10
step6850 | loss: 0.12467602640390396 | dt: 2001.84ms | tok/sec:  2046.12
step6860 | loss: 0.07351301610469818 | dt: 2008.32ms | tok/sec:  2039.52
step6870 | loss: 0.10613827407360077 | dt: 2040.12ms | tok/sec:  2007.72
step6880 | loss: 0.11000793427228928 | dt: 2015.84ms | tok/sec:  2031.91
step6890 | loss: 0.1365775167942047 | dt: 2008.23ms | tok/sec:  2039.61
step6900 | loss: 0.11174432188272476 | dt: 2019.46ms | tok/sec:  2028.27
step6910 | loss: 0.12290610373020172 | dt: 2007.40ms | tok/sec:  2040.45
step6920 | loss: 0.1151120662689209 | dt: 2020.72ms | tok/sec:  2027.00
step6930 | loss: 0.07084999978542328 | dt: 2013.00ms | tok/sec:  2034.78
...
step8510 | loss: 0.07909495383501053 | dt: 2028.61ms | tok/sec:  2019.12
step8520 | loss: 0.0849442407488823 | dt: 2006.25ms | tok/sec:  2041.62
step8530 | loss: 0.1092962771654129 | dt: 2008.95ms | tok/sec:  2038.87
step8540 | loss: 0.09598061442375183 | dt: 2004.89ms | tok/sec:  2043.01
step8550 | loss: 0.06828293949365616 | dt: 2009.25ms | tok/sec:  2038.57
step8560 | loss: 0.09338357299566269 | dt: 1990.21ms | tok/sec:  2058.07
step8570 | loss: 0.09850452840328217 | dt: 2000.60ms | tok/sec:  2047.39
step8580 | loss: 0.05907107889652252 | dt: 2006.51ms | tok/sec:  2041.36
step8590 | loss: 0.06736958026885986 | dt: 2002.29ms | tok/sec:  2045.65
step8600 | loss: 0.08429346978664398 | dt: 1990.69ms | tok/sec:  2057.57
step8610 | loss: 0.08821049332618713 | dt: 1988.62ms | tok/sec:  2059.72
step8620 | loss: 0.08477599173784256 | dt: 2040.11ms | tok/sec:  2007.74
step8630 | loss: 0.07638192176818848 | dt: 2013.80ms | tok/sec:  2033.97
step8640 | loss: 0.09862242639064789 | dt: 1990.05ms | tok/sec:  2058.24
step8650 | loss: 0.09760825335979462 | dt: 1997.26ms | tok/sec:  2050.81
step8660 | loss: 0.08491989225149155 | dt: 2009.02ms | tok/sec:  2038.81
step8670 | loss: 0.09003115445375443 | dt: 1997.04ms | tok/sec:  2051.03
step8680 | loss: 0.07729699462652206 | dt: 2007.18ms | tok/sec:  2040.67
step8690 | loss: 0.06784941256046295 | dt: 2019.37ms | tok/sec:  2028.35
step8700 | loss: 0.11659019440412521 | dt: 2007.33ms | tok/sec:  2040.52
step8710 | loss: 0.10317201167345047 | dt: 1996.44ms | tok/sec:  2051.65
...
step9810 | loss: 0.08238204568624496 | dt: 2032.77ms | tok/sec:  2014.99
step9820 | loss: 0.053803637623786926 | dt: 2007.62ms | tok/sec:  2040.22
step9830 | loss: 0.07872576266527176 | dt: 2015.56ms | tok/sec:  2032.19
step9840 | loss: 0.09004605561494827 | dt: 1986.45ms | tok/sec:  2061.97
step9850 | loss: 0.074969582259655 | dt: 1993.17ms | tok/sec:  2055.02
step9860 | loss: 0.0846974104642868 | dt: 2012.67ms | tok/sec:  2035.11
step9870 | loss: 0.07111480832099915 | dt: 1991.37ms | tok/sec:  2056.88
step9880 | loss: 0.08980842679738998 | dt: 2001.64ms | tok/sec:  2046.33
step9890 | loss: 0.07366511970758438 | dt: 1998.16ms | tok/sec:  2049.88
step9900 | loss: 0.07845361530780792 | dt: 2007.53ms | tok/sec:  2040.31
step9910 | loss: 0.07817139476537704 | dt: 1998.79ms | tok/sec:  2049.24
step9920 | loss: 0.08088140934705734 | dt: 1992.25ms | tok/sec:  2055.97
step9930 | loss: 0.08332135528326035 | dt: 1999.77ms | tok/sec:  2048.24
step9940 | loss: 0.06706588715314865 | dt: 2004.64ms | tok/sec:  2043.26
step9950 | loss: 0.07309224456548691 | dt: 2319.89ms | tok/sec:  1765.60
step9960 | loss: 0.09003520756959915 | dt: 2006.13ms | tok/sec:  2041.75
step9970 | loss: 0.0811273455619812 | dt: 2030.07ms | tok/sec:  2017.66
step9980 | loss: 0.08933334052562714 | dt: 1998.28ms | tok/sec:  2049.76
step9990 | loss: 0.05730853229761124 | dt: 2011.85ms | tok/sec:  2035.94
step10000 | loss: 0.05412352457642555 | dt: 2016.48ms | tok/sec:  2031.26
Checkpoint saved at checkpoints/checkpoint_10000.pth
Validation: (Step 10000), Generated text: Kohli comes forward and 19) [4. On the decision as well. On the decision as well. On the decision as well. On the decision as well. On the decision as well. On the decision as well. On the decision as well. On the
Reached maximum training steps.
```

## Sample outputs

```
1. 
Input:  Kohli comes forward
Output:  Kohli comes forward and lofts over long-off. High and handsome

KXIP 1st Inns
6.5 overs
Chawla to Saha, SIX, Amla welcomes 'em with the googly out of the park

========================


2. 
Input:  over the rope and
Output:  over the rope and the first attacking fence

RCB 1st Inns
13.4 overs
ra Arcy Short to de Villiers, out Caught by Unadkat!! Failed back into the pull shot, strikes it seemed

========================


3. 
Input:  in the air
Output:  in the air, he did well to have mistimed it big when his executed is under pressure. Fullish ball, Ankit Soni c Pant biraj 37(32) [4s-3 6s-2]


========================


4. 
Input:  what a brilliant shot
Output:  what a brilliant shot from final into the night sky, but unfortunately, it wasn't proper nice. Vohra c Kohli b Washington Sundar 28(13) [4s-4 6s-1]

RCB 1

========================


5. 
Input:  just missed by a
Output:  just missed by a 95 win for 8 in his 6. The crowd mighty Viratia!! That's the opening Good Archer, who wants to swing this one out to an ways to the tumble, we informs us how hard it plays.

========================

```

## 📂 Repository Structure

```
s15erav3/
│── checkpoints/       # Stores model checkpoints
│── config.yaml        # Config file with hyperparameters
│── model.py           # Model architecture including MLHA & MoE
│── train.py           # Training script
│── requirements.txt   # Dependencies
│── README.md          # Project documentation
│── utils.py           # Helper functions
│── input.txt          # Training dataset
```

## 🔥 Model Architecture

### **1️⃣ Multi-Head Latent Attention (MLHA)**
MLHA improves traditional **Multi-Head Attention (MHA)** by:
- Reducing computational complexity while preserving long-range dependencies.
- Selectively attending to latent features instead of raw inputs.
- Enhancing model interpretability and efficiency.

### **2️⃣ Mixture of Experts (MoE) with Load Balancing**
MoE optimizes model computation by activating a subset of expert layers per input token. Key components:
- **Top-K Routing**: Selects the most relevant experts per token dynamically.
- **Load Balancing Loss**: Ensures even distribution of workload across experts.
- **Sparse Activation**: Enhances computational efficiency while maintaining accuracy.

### **3️⃣ Loss-less Load Balancing**
Ensures that:
- Input tokens are evenly distributed across available experts.
- No expert is overloaded or underutilized.
- Smooth training convergence with balanced gradient updates.

## 🛠 Installation

1️⃣ Clone this repository:
```bash
git clone https://github.com/piygr/s15erav3.git
cd s15erav3
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Training

1️⃣ **Modify `config.yaml`** to set hyperparameters such as `learning_rate`, `batch_size`, and expert configurations.

2️⃣ **Start Training**:
```bash
python train.py
```

## 🏗 Future Enhancements

- 🔹 Integration of **Sparse MoE Variants** (e.g., Switch Transformers)
- 🔹 Advanced Load Balancing Mechanisms for Scalable MoE
- 🔹 Optimization of MLHA for Large Transformer Models


## 📜 License

This project is open-source and licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

🚀 **Now you're ready to train and experiment with MLHA and MoE!** If you have any questions, feel free to open an issue in the repository. 🔥

