# CLAUDE.md

usd 파일을 불러와서 학습시킨다.
isaacLab 3.0을 사용하되, isaacSim은 사용하지 않는다.
최대 목표는 160mm, 300mm 계단을 오르는 것이다.
구현 방식은, 
우선 걷기, 계단 오르기 두 task로 진행한다.
nivid 공식 SO101 arm의 Tutorial를 따라하여 증류 방식을 사용한다. (teacher 학습 후, student 증류)
