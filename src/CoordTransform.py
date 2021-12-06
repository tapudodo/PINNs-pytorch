# -*- coding: utf-8 -*-
import numpy as np

def Square2Circle(xi,eta,radius):
    r2 = radius*radius
    x = np.multiply(xi/radius,np.sqrt(r2-0.5*eta**2 ) ) # element wise multiplication
    y = np.multiply(eta/radius,np.sqrt(r2-0.5*xi**2 ) )
    return x,y


def Circle2Square(x,y,radius):
    r2 = radius*radius
    x2 = np.multiply(x,x)
    y2 = np.multiply(y,y)
    xi = 0.5*np.sqrt(2*r2+x2-y2+2*np.sqrt(2)*radius*x)- 0.5*np.sqrt(2*r2+x2-y2-2*np.sqrt(2)*radius*x)
    eta = 0.5*np.sqrt(2*r2-x2+y2+2*np.sqrt(2)*radius*y)- 0.5*np.sqrt(2*r2-x2+y2-2*np.sqrt(2)*radius*y)
    return xi, eta

def dxi_eta(x,y,radius):
    r2 = radius*radius
    x2 = np.multiply(x,x)
    y2 = np.multiply(y,y)
    sqrt2r = np.sqrt(2)*radius
    xi_denominator1 = 0.5*(2*r2+x2-y2+2*sqrt2r*x)
    xi_denominator2 = 0.5*(2*r2+x2-y2-2*sqrt2r*x)
    eta_denominator1 = 0.5*(2*r2-x2+y2+2*sqrt2r*y)
    eta_denominator2 = 0.5*(2*r2-x2+y2-2*sqrt2r*y)
    dxidx = np.divide(x+sqrt2r,xi_denominator1)-np.divide(x-sqrt2r,xi_denominator2)
    dxidy = np.divide(-y,xi_denominator1)+np.divide(y,xi_denominator2)
    detadx = np.divide(y+sqrt2r,eta_denominator1)-np.divide(y-sqrt2r,eta_denominator2)
    detady = np.divide(-x,eta_denominator1)+np.divide(x,eta_denominator2)
    return (dxidx, dxidy, detadx, detady)