import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import ProgressBarCallback
import time

# Pusher ortamını oluşturun
env = gym.make('Pusher-v4', render_mode="rgb_array")

# Modeli başlatın ve buffer boyutunu ayarlayın
model = SAC('MlpPolicy', env, buffer_size=100000, verbose=1)

# İlerlemenin her adımda gösterilmesi için ProgressBarCallback kullanın
callback = ProgressBarCallback()

# Eğitim
print("Eğitim başlıyor...")
model.learn(total_timesteps=5000, callback=callback)  # Eğitim adım sayısını isteğinize göre artırabilirsiniz

# Eğitim tamamlandığında modeli kaydedin
model.save("sac_pusher")
print("Eğitim tamamlandı ve model kaydedildi.")

# Modeli yükleyin
model = SAC.load("sac_pusher")

# Test döngüsü
print("Model test ediliyor...")
obs, _ = env.reset()

for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()  # render() çağrısı görselleştirmeyi ekrana getirir
    time.sleep(0.01)  # Görselleştirmeyi yavaşlatmak için isteğe bağlı olarak ekleyin
    if done:
        obs, _ = env.reset()

env.close()
print("Test tamamlandı.")
