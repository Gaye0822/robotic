import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import ProgressBarCallback
import time

# CarRacing ortamını render_mode ile oluşturun
env = gym.make('CarRacing-v2', render_mode="human")

# Modeli başlatın ve buffer boyutunu ayarlayın
model = SAC('CnnPolicy', env, buffer_size=100000, verbose=1)

# İlerlemenin her adımda gösterilmesi için ProgressBarCallback kullanın
callback = ProgressBarCallback()

# Eğitim
print("Eğitim başlıyor...")
model.learn(total_timesteps=10000, callback=callback)  # Eğitim adım sayısını daha büyük yapabilirsiniz

# Eğitim tamamlandığında modeli kaydedin
model.save("sac_car_racing")
print("Eğitim tamamlandı ve model kaydedildi.")

# Modeli yükleyin
model = SAC.load("sac_car_racing")

# Test döngüsü
print("Model test ediliyor...")
obs, _ = env.reset()

for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()  # render() çağrısı görselleştirmeyi ekrana getirir
    time.sleep(0.01)  # Görselleştirmeyi yavaşlatmak için isteğe bağlı olarak ekleyin
    if done:
        obs, _ = env.reset()

env.close()
print("Test tamamlandı.")
