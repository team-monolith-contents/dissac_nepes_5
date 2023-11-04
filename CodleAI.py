from ipywidgets import IntText, IntSlider, Button, HBox, Output, interactive, VBox
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import seaborn as sns
import pandas as pd
import urllib.request
import gym

class AIProject1():
    def __init__(self):
        self.__a = IntSlider(description='기울기 :', min=0, max=30) # a 변수의 값을 조절할 수 있는 IntSlider를 만들어줍니다.
        self.__b = IntSlider(description='y절편 :', min=-300, max=300) # b 변수의 값을 조절할 수 있는 IntSlider를 만들어줍니다.
        self.__out = Output() # ipywidget의 output을 만들어줍니다.

        self.__data_x = np.array([12, 20, 30, 56])
        self.__data_y = np.array([150, 208, 500, 875])

        display(self.__a,self.__b,self.__out) # a,b,out을 출력단에 표시해줍니다.

        with self.__out:
            self.__fig = plt.figure()
            self.__draw(self.__fig, self.__a.value, self.__b.value)

        self.__a.observe(self.__on_a_value_change, names='value') # a가 변화할 때 마다 실행 될 콜백함수를 등록합니다.
        self.__b.observe(self.__on_b_value_change, names='value') # b가 변화할 때 마다 실행 될 콜백함수를 등록합니다.
    
    # loss 함수 입니다.
    def __loss(self, a, b, data_x, data_y):
        return sum(((a*data_x + b)- data_y)**2)/ len(data_x)

    # ipywidget의 output을 그려주는 함수입니다.
    def __draw(self, fig,a,b):

        clear_output(wait=True) # ipywidget에 포함된 함수로 output을 초기화 시켜주는 함수입니다.
        margin = 5

        # 아래의 코드로 출력되는 결과는 모두 ipywidget output으로 보내줍니다.
        plt.scatter(self.__data_x, self.__data_y)
        plt.plot([min(self.__data_x)-margin, max(self.__data_x)+margin],
                [a*min(self.__data_x)+b, a*max(self.__data_x)+b], c = 'r')
        plt.xlim(0, 70)
        plt.ylim(0, 1200)
        plt.show()

        sign = "+" if b >= 0 else "-"
        
        print(f"{a} * x {sign} {abs(b)}")
        print("평균 제곱 오차 :", self.__loss(a, b, self.__data_x, self.__data_y))
        print("[도전] 평균 제곱 오차를 2000 아래로 줄여보세요!")

    # a 값이 변경될 때 실행될 콜백함수입니다.
    def __on_a_value_change(self, change):
        with self.__out: # with out: block에 있는 코드들은 출력값을 ipywidget output으로 보내줍니다.
            self.__draw(self.__fig,change['new'], self.__b.value)

    # b 값이 변경될 때 실행될 콜백함수입니다.
    def __on_b_value_change(self, change):
        with self.__out:
            self.__draw(self.__fig, self.__a.value, change['new'])

class AIProject2():
    def __init__(self):
        # image 다운로드
        url = "https://tmn-bucket-materials-all.s3.ap-northeast-2.amazonaws.com/data/COVID19-Xray.jpeg"
        urllib.request.urlretrieve(url, "COVID19-Xray.jpeg")
        
        # X-ray 이미지를 불러옵니다.
        self.__xray_image = cv2.imread('COVID19-Xray.jpeg', cv2.IMREAD_GRAYSCALE)
        self.__xray_image = cv2.resize(self.__xray_image, dsize=(0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
        self.__df_image = pd.DataFrame(self.__xray_image)
        
        w = interactive(self.__check_gray_image_value, 
                vertical=IntSlider(min=0, max=self.__xray_image.shape[0]-15, value=10), 
                horizontal=IntSlider(min=0, max=self.__xray_image.shape[1]-15, value=30)
               )
        
        display(VBox([w.children[1], w.children[0], w.children[-1]]))

    def __check_gray_image_value(self,vertical,horizontal):
        rs, re = vertical, vertical + 15
        cs, ce = horizontal, horizontal + 15
        # 이미지를 출력합니다.
        fig = plt.figure(figsize=(15, 6))
        fig.add_subplot(1, 2, 1)
        plt.xlim(0, self.__xray_image.shape[1]-1)
        plt.ylim(self.__xray_image.shape[0]-1, 0)
        plt.imshow(self.__xray_image, cmap='gray')
        # 초록박스를 표시합니다.
        plt.plot([cs, ce, ce, cs, cs], [rs, rs, re, re, rs], color="darkgreen", linewidth=1.5)
        # 화소값을  출력합니다.
        fig.add_subplot(1, 2, 2)  
        sns.heatmap(self.__df_image.iloc[rs:re, cs:ce], vmin=0, vmax=255, annot=True, fmt='d', 
                    cmap=sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True))
        plt.show()

class AIProject3():
    def __init__(self, map):
        self.__env = gym.make('FrozenLake-v1', desc=map, render_mode='rgb_array', is_slippery=False)
        self.__env.reset()
        
        self.__action_size = self.__env.action_space.n
        self.__observation_size = self.__env.observation_space.n
        self.__Q = np.zeros([self.__observation_size, self.__action_size])
        
        self.__epsilon = 1
        self.__alpha = 0.95
        self.__gamma = 0.8
        self.__lamb = 0.9
        self.__epsilon_update = 1000
        self.__episode_rewards = []
        self.__show_interval = 1000
        self.__plot_interval = 1000
        
        self.__episode = IntSlider(
            value=0,
            min=0,
            max=100000,
            step=1,
            disabled = True
        )
        
        self.__N_episode = IntText(
            value=10000,
            description='학습횟수:',
            disabled=False
        )
        
        self.__learn_button = Button(
            description='학습하기',
            disabled=False
        )
        
        self.__test_button = Button(
            description='테스트하기',
            disabled=False
        )
            
        self.__out = Output()
        display(HBox([self.__learn_button, self.__N_episode, self.__episode]), self.__test_button, self.__out)
        self.__learn_button.on_click(self.__learn)
        self.__test_button.on_click(self.__test)

    def __epsilon_greedy(self, state):
        if np.random.rand() > self.__epsilon:
            return np.argmax(self.__Q[state, :])
        else:
            return self.__env.action_space.sample()
            
    def __learn(self, button):
        for _ in range(self.__N_episode.value):
            self.__env.reset()
            state = 0
            rewards = []
            done = False

            if self.__episode.value % self.__epsilon_update == 0:
                self.__epsilon *= self.__lamb

            if self.__episode.value % self.__plot_interval == 0:
                    self.__show()
                    with self.__out:
                        print(self.__Q)

            while not done:
                action = self.__epsilon_greedy(state)
                next_state, reward, done, _, info = self.__env.step(action)
                self.__Q[state][action] += self.__alpha * (
                            reward + self.__gamma * self.__Q[next_state].max() - self.__Q[state][action])
                state = next_state
                rewards.append(reward)

                if self.__episode.value % self.__show_interval == 0:
                    self.__show()

            try:
                self.__episode_rewards[(self.__episode.value)//self.__plot_interval] += sum(rewards)/len(rewards)/self.__plot_interval
            except:
                self.__episode_rewards.append(sum(rewards)/len(rewards)/self.__plot_interval)

            self.__episode.value += 1
                
    def __show(self):
        with self.__out:
            screen = self.__env.render()
            clear_output(wait=True)
            plt.clf()
            
            plt.subplot(1,2,1)
            plt.plot(np.arange(len(self.__episode_rewards)+1) * self.__plot_interval, [0] + (self.__episode_rewards), marker = 'o')
            plt.subplot(1,2,2)
            plt.imshow(screen) # screen 배열을 이미지로 출력합니다.
            
            plt.show()

    def __test(self, button):
        with self.__out:
            self.__env.reset()
            cur_state=0
            done = False
            self.__show()
            while not done:
                next_state, reward, done, info,_ = self.__env.step(self.__Q[cur_state].argmax())
                cur_state = next_state
                with self.__out:
                    print("테스트중입니다....")
                self.__show()