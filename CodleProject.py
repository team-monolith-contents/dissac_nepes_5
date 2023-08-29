from ipywidgets import Image
from ipycanvas import Canvas
import ipywidgets as widgets
from IPython.display import display, clear_output

import random

from ipywidgets import Image
from ipycanvas import Canvas
import ipywidgets as widgets
from IPython.display import display, clear_output

class BaseballPlay():
    def __init__(self):
        
        def make_answer():
            while True:
                answer = random.randint(100,999)
                answer = str(answer)
                if answer[0] != answer[1] and answer[0] != answer[2] and answer[1] != answer[2]:
                    return answer

        def strike_ball(player, answer):
            s = 0
            b = 0
            for j in range(len(player)):
                if player[j] == answer[j]:
                    s += 1
                elif player[j] in answer:
                    b += 1
            return s, b

        def judge(s, b):
            if s == 3:
                return True
            else:
                return False

        def add_history(history, result, player, s,b):
            history.append(player)
            result.append([s,b])
            return history, result

        self.__make_answer = make_answer
        self.__strike_ball = strike_ball
        self.__judge = judge
        self.__add_history = add_history
        
        self.__canvas = Canvas(width=1280, height=720)
        self.__canvas.draw_image(Image.from_file("baseball.png"), 0,0)
        
        self.__inning = 0
        self.__player = None
        self.__history = []
        self.__result = []
        self.__flag = 0
        
        self.__play_button = widgets.Button(
            description='PLAY',
            disabled=False,
            button_style='',
            tooltip='PLAY'
        )
        self.__player_answer = widgets.BoundedIntText(
            min=0,
            max=999,
            step=1,
            description='정답 입력:',
            disabled=False
        )
        self.__output = widgets.Output()
        display(self.__play_button, self.__player_answer, self.__canvas, self.__output)
        
        self.__player_answer.observe(self.__on_player_answer_change, names='value')
        self.__play_button.on_click(self.__play)
            
    def __on_player_answer_change(self, change):
        with self.__output:
            self.__player = f"{change['new']:03}"
            if self.__player and self.__check_input(self.__player):
                s,b = self.__strike_ball(self.__player, self.__answer)
                self.__history, self.__result = self.__add_history(self.__history, self.__result, self.__player, s, b)
                if self.__judge(s,b):
                    self.__inning = 10
                    self.__flag = 1
                    self.__show_end()
                else:
                    self.__inning += 1
                    self.__show_play()
                    
            self.__canvas.fill_style = "white"
            self.__canvas.font = "200px serif"
            self.__canvas.fill_text(f"{self.__player}", 200, 420)
            
            if self.__inning == 10:
                self.__show_end()
            
    def __check_input(self, num):
        return num[0] != num[1] and num[0] != num[2] and num[1] != num[2]
    
    def __show_play(self):
        with self.__output:
            self.__canvas.clear()
            self.__canvas.draw_image(Image.from_file("background.png"), 0,0)
            self.__canvas.fill_style = "white"
            self.__canvas.font = "40px serif"
            self.__canvas.fill_text(f"{self.__inning} 번째 이닝({self.__inning}/9)", 110, 75)
            
            for idx, his in enumerate(self.__history):
                self.__canvas.fill_style = "black"
                self.__canvas.font = "40px serif"
                self.__canvas.fill_text(f"{his}", 910, idx*58 + 190)
            
            for idx, res in enumerate(self.__result):
                self.__canvas.fill_style = "black"
                self.__canvas.font = "40px serif"
                self.__canvas.fill_text(f"S:{res[0]}   B:{res[1]}", 1000, idx*58 + 190)
            
    def __show_end(self):
        self.__canvas.clear()
        if self.__flag:
            self.__canvas.draw_image(Image.from_file("end_success.png"), 0,0)
        else:
            self.__canvas.draw_image(Image.from_file("end_fail.png"), 0,0)
    
    def __play(self, button):
        self.__inning = 1
        self.__answer = self.__make_answer()
        self.__player = "000"
        self.__history = []
        self.__result = []
        self.__flag = 0
        self.__player_answer.value = 0
        self.__show_play()
        self.__play_button.description = 'REPLAY'
        self.__play_button.tooltip = 'REPLAY'
        
class BaseBallProject():
    def __init__(self, make_answer, strike_ball, judge, add_history):
        self.__make_answer = make_answer
        self.__strike_ball = strike_ball
        self.__judge = judge
        self.__add_history = add_history
        
        self.__canvas = Canvas(width=1280, height=720)
        self.__canvas.draw_image(Image.from_file("baseball.png"), 0,0)
        
        self.__inning = 0
        self.__player = None
        self.__history = []
        self.__result = []
        self.__flag = 0
        
        self.__play_button = widgets.Button(
            description='PLAY',
            disabled=False,
            button_style='',
            tooltip='PLAY'
        )
        self.__player_answer = widgets.BoundedIntText(
            min=0,
            max=999,
            step=1,
            description='정답 입력:',
            disabled=False
        )
        self.__output = widgets.Output()
        display(self.__play_button, self.__player_answer, self.__canvas, self.__output)
        
        self.__player_answer.observe(self.__on_player_answer_change, names='value')
        self.__play_button.on_click(self.__play)
            
    def __on_player_answer_change(self, change):
        with self.__output:
            self.__player = f"{change['new']:03}"
            if self.__player and self.__check_input(self.__player):
                s,b = self.__strike_ball(self.__player, self.__answer)
                self.__history, self.__result = self.__add_history(self.__history, self.__result, self.__player, s, b)
                if self.__judge(s,b):
                    self.__inning = 10
                    self.__flag = 1
                    self.__show_end()
                else:
                    self.__inning += 1
                    self.__show_play()
                    
            self.__canvas.fill_style = "white"
            self.__canvas.font = "200px serif"
            self.__canvas.fill_text(f"{self.__player}", 200, 420)
            
            if self.__inning == 10:
                self.__show_end()
            
    def __check_input(self, num):
        return num[0] != num[1] and num[0] != num[2] and num[1] != num[2]
    
    def __show_play(self):
        with self.__output:
            self.__canvas.clear()
            self.__canvas.draw_image(Image.from_file("background.png"), 0,0)
            self.__canvas.fill_style = "white"
            self.__canvas.font = "40px serif"
            self.__canvas.fill_text(f"{self.__inning} 번째 이닝({self.__inning}/9)", 110, 75)
            
            for idx, his in enumerate(self.__history):
                self.__canvas.fill_style = "black"
                self.__canvas.font = "40px serif"
                self.__canvas.fill_text(f"{his}", 910, idx*58 + 190)
            
            for idx, res in enumerate(self.__result):
                self.__canvas.fill_style = "black"
                self.__canvas.font = "40px serif"
                self.__canvas.fill_text(f"S:{res[0]}   B:{res[1]}", 1000, idx*58 + 190)
            
    def __show_end(self):
        self.__canvas.clear()
        if self.__flag:
            self.__canvas.draw_image(Image.from_file("end_success.png"), 0,0)
        else:
            self.__canvas.draw_image(Image.from_file("end_fail.png"), 0,0)
    
    def __play(self, button):
        self.__inning = 1
        self.__answer = self.__make_answer()
        self.__player = "000"
        self.__history = []
        self.__result = []
        self.__flag = 0
        self.__player_answer.value = 0
        self.__show_play()
        self.__play_button.description = 'REPLAY'
        self.__play_button.tooltip = 'REPLAY'
        
class SortingProject():
    def __init__(self, arr):
        self.__ending(arr)

    def __ending (self, O0O00OOO000000000 ):#line:1
        O0OOO0000OO000O0O ={185 :"철수",165 :"영희",180 :"동휘",175 :"민수",170 :"진호"}#line:8
        OO0O0OO0O0O00OOOO =list (O0OOO0000OO000O0O .keys ())#line:9
        OO0O0OO0O0O00OOOO .sort ()#line:10
        import time #line:12
        if (type (O0O00OOO000000000 )!=list ):#line:13
            print ("애들 다 어디갔어? 얘들아??")#line:14
            return #line:15 
        time .sleep (0.75 )#line:16
        if (len (O0O00OOO000000000 )<5 ):#line:18
            print ("뭐야! 왜 사람이 5명이 아니지...? 나머지 어디갔어! 불러와!")#line:19
            return #line:20
        elif (len (O0O00OOO000000000 )>5 ):#line:21
            print ("뭐야! 왜 사람이 5명이 아니지...? 왜 이렇게 많아! 나가봐!")#line:22
            return #line:23
        else :#line:24
            print ("좋아 일단 5명은 모였고...")#line:25
        time .sleep (0.75 )#line:26
        O0OOOO0OO0000O0OO =set (O0O00OOO000000000 )#line:28
        OOOOOO0OO00OOO00O =set (O0OOO0000OO000O0O .keys ())#line:29
        OOOO0OOO0O0OOO0O0 =OOOOOO0OO00OOO00O .difference (O0OOOO0OO0000O0OO )#line:30
        if (len (OOOO0OOO0O0OOO0O0 )==0 ):#line:31
            print (f"{', '.join(O0OOO0000OO000O0O.values())} 다 모였네")#line:32
        else :#line:33
            OOO0O00O00OO0OOOO =[]#line:34
            for OOOO00OOO00000O0O in OOOO0OOO0O0OOO0O0 :#line:35
                OOO0O00O00OO0OOOO .append (O0OOO0000OO000O0O [OOOO00OOO00000O0O ])#line:36
            print (f"엥 저기..{', '.join(OOO0O00O00OO0OOOO)}가 아닌데..? 엉뚱한애 데려오지 말고 가서 찾아와!")#line:37
            return #line:38
        time .sleep (0.75 )#line:39
        if (O0O00OOO000000000 [4 ]==OO0O0OO0O0O00OOOO [0 ]and O0O00OOO000000000 [3 ]==OO0O0OO0O0O00OOOO [1 ]and O0O00OOO000000000 [2 ]==OO0O0OO0O0O00OOOO [2 ]and O0O00OOO000000000 [1 ]==OO0O0OO0O0O00OOOO [3 ]and O0O00OOO000000000 [0 ]==OO0O0OO0O0O00OOOO [4 ]):#line:41
            print ("야 거꾸로 섰어 ㅋㅋㅋㅋㅋ 카메라 이쪽이니깐 반대로 서라!")#line:42
            return #line:43
        for OOOO0O0000OO0OOOO in range (5 ):#line:45
            for O0000OO000O00O00O in range (OOOO0O0000OO0OOOO +1 ,5 ):#line:46
                if O0O00OOO000000000 [OOOO0O0000OO0OOOO ]>O0O00OOO000000000 [O0000OO000O00O00O ]:#line:47
                    print (f"야야 저기 {O0OOO0000OO000O0O[O0O00OOO000000000[OOOO0O0000OO0OOOO]]} {O0OOO0000OO000O0O[O0O00OOO000000000[O0000OO000O00O00O]]}한테 가린다 다시 한번 서봐!")#line:48
                    return #line:49
        print ("자 그럼 찍는다! 하나...둘..셋!")#line:50
        time .sleep (0.5 )#line:51
        print ("하나 더 찍을게! 하나...둘..셋!")#line:52
        time .sleep (0.5 )#line:53
        print ("고생 많았다 얘들아~ 졸업 축하한다:)")#line:54

    