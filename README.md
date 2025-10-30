# CS885 Assignment 2
## Assignment 2-1
* **檔案**: `CS885_Assignment_2-1.ipynb`
## Assignment 2-2
### Cartpole:

1. **REINFORCE**:
* Eval score: 200.00 ± 0.00
學習曲線穩定上升，且最終完成了任務，但是學習曲線比較震盪，這是因為使用的是最基礎的Policy Gradient演算法，變異性(variance)較高所導致的。
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/abc88d65-3d42-4493-82b5-ac0c81dd0d3e" />


2. **REINFORCE_Baseline**:
* Eval score: 200.00 ± 0.00(POLICY_TRAIN_ITERS = 1)
由於使用了價值網路來計算優勢函數(Advantage)，因此降低了variance，使得學習速度較REINFORCE快，震盪也比較小。
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/79edcbe8-46e9-4ff1-87e5-566d3b290504" />

* Eval score: 9.50 ± 0.67(POLICY_TRAIN_ITERS = 10)
嘗試將POLICY_TRAIN_ITERS從1增加到10，結果出現了學習崩潰 (catastrophic forgetting)的現象，這是因為REINFORCE是on policy的演算法，當我們用同一批數據連續更新網路10次，會導致使用了過去的資料來更新新的policy，破壞了 On-Policy 的基本假設。
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/c95176aa-472f-481a-a4de-053a1b514231" />



3. **PPO**:
* Eval score: 130.90 ± 41.07
PPO是三者中表現最好的，因為他只需100個episode就能收斂，且過程相當穩定，沒有出現震盪。
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/32725a7b-0c27-478a-86b0-8d9c80b0e584" />


### Mountain_car:
在Mountain_car的環境中，所有演算法都失敗了，學習曲線都是一條平線，這是因為環境中的稀疏獎勵導致的，每個時間步中，agent都會得到-1的懲罰，只有在成功到達終點時才會獲得0的獎勵，這導致agent所收到的回饋都是負面的，無法藉此區分比較好的動作，因此無法學到有用的策略。
1. **REINFORCE**:
* Height achieved: -104.13 ± 2.69
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/28e68e45-2639-4310-b8cc-49e35b41b079" />


2. **REINFORCE_Baseline**:
* Height achieved: -102.71 ± 2.49
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/70854e3d-6d26-4467-af8f-351490e74ba2" />


3. **PPO**:
* Height achieved: -104.75 ± 1.17
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/1b6d5850-811b-4a6e-8359-e5fb91439e55" />


### Mountain_car_mod:
在使用修改過的reward後，可看出三種演算法都能成功學會如何解決問題，將獎勵從「每步 -1」改為「當前車子的高度」，使得agent每做一個動作，都能得到一個即時的、有意義的回饋。
1. **REINFORCE**:
* Height achieved: -100.55 ± 2.23
雖然表現較未修改的版本好，但是曲線相當不穩定，最終高度也最低。
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/1fa22a12-9f4b-49d0-b6b2-785232071a7d" />


2. **REINFORCE_Baseline**:
* Height achieved: -85.05 ± 2.06
可看出學習曲線有明顯上升趨勢，學習過程也比REINFORCE穩定。
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/ad9a087c-cee4-4aae-a47c-df2b49f6963a" />


3. **PPO**:
* Height achieved: -74.99 ± 1.22
在三者中表現最好，不但取得最好的成績，收斂速度也是最快的。
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/9d29eca9-2b84-4843-bc17-0bf9c7378429" />
