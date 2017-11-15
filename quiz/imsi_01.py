a=100
b=0

while b<=a:
  b=b+1
  if a % b !=0: continue
  print("b=", b) #====b는 a의 인수

  c=0
  while c<b:
    c=c+1
    if b%c ==0:
      print("c=", c) #=====c는 b의 인수
