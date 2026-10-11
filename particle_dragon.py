#!/usr/bin/env python3
"""Render an 8-second particle transformation from an input painting.
Usage: python particle_dragon.py input.png preview.mp4 --preview
Requires Pillow, numpy, imageio-ffmpeg (or ffmpeg executable).
"""
import argparse, subprocess, os, tempfile
import numpy as np
from PIL import Image, ImageFilter

def ease(x): return x*x*(3-2*x)
def render(inp,out,fps=24,preview=False):
 im=Image.open(inp).convert('RGB'); w,h=im.size
 scale=min(1.0, 720/max(w,h)) if preview else 1.0
 W,H=max(2,int(w*scale)//2*2),max(2,int(h*scale)//2*2)
 im=im.resize((W,H),Image.Resampling.LANCZOS); a=np.asarray(im).astype(np.float32)
 # foreground: chroma departure from paper-white
 lum=a.mean(2); mask=(lum<248) | (a.max(2)-a.min(2)>22)
 ys,xs=np.where(mask); rng=np.random.default_rng(7)
 n=min(26000,len(xs)); idx=rng.choice(len(xs),n,replace=False)
 x=xs[idx].astype(float); y=ys[idx].astype(float); col=a[ys[idx],xs[idx]]/255
 # particle offsets along a smooth pseudo-flow field
 ang=np.sin(y*.035)+np.cos(x*.027); dx=np.cos(ang)*7; dy=np.sin(ang)*7
 tmp=tempfile.NamedTemporaryFile(suffix='.mp4',delete=False); tmp.close()
 cmd=['ffmpeg','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','18','-movflags','+faststart',out]
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 for k in range(8*fps):
  t=k/(8*fps-1); q=0 if t<.18 else (1 if t> .82 else ease((t-.18)/.64))
  # q=1 dispersed; q returns to 0 in final quarter
  if t>.68: q*=1-ease((t-.68)/.32)
  xx=x+dx*q*(1+0.25*np.sin(t*18+y*.01)); yy=y+dy*q*(1+0.25*np.cos(t*16+x*.01))
  frame=np.full((H,W,3),250,dtype=np.float32)
  if q<.08 or t<.12 or t>.88:
   frame=np.array(im,dtype=np.float32)
  else:
   xi=np.clip(xx.round().astype(int),0,W-1); yi=np.clip(yy.round().astype(int),0,H-1)
   frame[yi,xi]=np.maximum(frame[yi,xi],col*255)
   # soft glow layer
   glow=Image.fromarray(np.uint8(np.clip(frame,0,255))).filter(ImageFilter.GaussianBlur(radius=max(1,int(3*q))))
   frame=0.72*frame+0.28*np.asarray(glow)
  p.stdin.write(np.uint8(np.clip(frame,0,255)).tobytes())
 p.stdin.close(); p.wait()
 if p.returncode: raise SystemExit(p.returncode)
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('output'); ap.add_argument('--preview',action='store_true'); a=ap.parse_args(); render(a.input,a.output,12 if a.preview else 24,a.preview)
