// Copyright 2016 constrained Developers
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate sprs;
extern crate num;
extern crate rand;

use sprs::sparse::compressed::SpMatView;
use sprs::sparse::CompressedStorage;

use sprs::{CsVecView, CsMatView};

use sprs::{CsVec, CsMat, CsVecOwned};

use sprs::sparse::linalg::cholesky::LdlNumeric;
use sprs::sparse::vec::SparseIterTools;

use num::Num;
use std::default::Default;
use std::ops::Deref;
use std::fmt::{Debug};
use num::{One, Zero, Float};

trait VecLike {
    type Item;
    fn size(&self) -> usize;
    fn at(&self, i: usize) -> Self::Item;
}

impl<N: Clone> VecLike for [N] {
    type Item = N;
    fn size(&self) -> usize { self.len() }
    fn at(&self, i: usize) -> N { self[i].clone() }
}

impl<'a, N: Clone> VecLike for &'a [N] {
    type Item = N;
    fn size(&self) -> usize { self.len() }
    fn at(&self, i: usize) -> N { self[i].clone() }
}

struct Cropped<V: VecLike> {
    v: V,
    source_start: usize,
    target_start: usize,
    target_end: usize,
    target_len: usize
}

fn crop<V: VecLike>(v: V, source_start: usize, target_start: usize, target_end: usize, target_len: usize) -> Cropped<V> {
    Cropped {
        v: v,
        source_start: source_start,
        target_start: target_start,
        target_end: target_end,
        target_len: target_len
    }
}

impl<V: VecLike> VecLike for Cropped<V> where V::Item: Zero {
    type Item = V::Item;
    fn size(&self) -> usize { self.target_len }
    fn at(&self, i: usize) -> V::Item {
        if i >= self.target_start && i < self.target_end {
            self.v.at(i - self.target_start + self.source_start)
        } else {
            V::Item::zero()
        }
    }
}

impl<N, IS, DS> VecLike for CsVec<N, IS, DS>
where N: Zero + Copy, IS: Deref<Target=[usize]>, DS: Deref<Target=[N]>
{
    type Item = N;
    fn size(&self) -> usize { self.dim() }
    fn at(&self, i: usize) -> N { self.at(i).unwrap_or_else(|| N::zero()) }
}

impl<N: Clone> VecLike for Vec<N> {
    type Item = N;
    fn size(&self) -> usize { self.len() }
    fn at(&self, i: usize) -> N { self[i].clone() }
}

trait MatrixLike {
    type Item;
    fn size(&self) -> (usize, usize);
    fn at(&self, i: usize, j: usize) -> Self::Item;
}

impl<N, PS, IS, DS> MatrixLike for CsMat<N, PS, IS, DS>
where N: Copy + Zero, PS: Deref<Target=[usize]>, IS: Deref<Target=[usize]>, DS: Deref<Target=[N]>
{
    type Item = N;
    fn size(&self) -> (usize, usize) { (self.rows(), self.cols()) }
    fn at(&self, i: usize, j: usize) -> N { self.at(&(i, j)).unwrap_or_else(|| N::zero()) }
}

fn per_element<N, V: VecLike + ?Sized, F: FnMut(V::Item) -> N>(v: &V, mut f: F) -> Vec<N> {
    let mut r = Vec::with_capacity(v.size());
    for i in 0..v.size() {
        r.push(f(v.at(i)));
    }
    r
}
fn combine_vec_2<N, V1: VecLike + ?Sized, V2: VecLike + ?Sized, F: FnMut(V1::Item, V2::Item) -> N>(v1: &V1, v2: &V2, mut f: F) -> Vec<N> {
    assert_eq!(v1.size(), v2.size());
    let mut r = Vec::with_capacity(v1.size());
    for i in 0..v1.size() {
        r.push(f(v1.at(i), v2.at(i)));
    }
    r
}

fn from_vec<N: Copy>(v: Vec<N>) -> CsVecOwned<N> {
    let mut indices = Vec::with_capacity(v.len());
    for i in 0..v.len() {
        indices.push(i);
    }
    CsVec::new_owned(v.len(), indices, v).unwrap()
}

fn from_iter<N: Copy, I: Iterator<Item=(usize, N)>>(itr: I, len: Option<usize>) -> CsVecOwned<N> {
    let min = itr.size_hint().0;
    let mut indices = Vec::with_capacity(min);
    let mut data = Vec::with_capacity(min);
    for (i, d) in itr {
        indices.push(i);
        data.push(d);
    }
    CsVec::new_owned(len.unwrap_or_else(|| indices[indices.len()-1] + 1), indices, data).unwrap()
}

fn rand_vec<N: rand::Rand + Zero + PartialEq>(n: usize) -> Vec<N> {
    let mut r = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = rand::random::<N>();
        while v == N::zero() {
            v = rand::random::<N>();
        }
        r.push(v);
    }
    r
}

#[derive(Debug)]
pub enum SolverError {
    NoFeasibleSolution,
    Unbounded
}
fn dot<N: Copy + Num>(a: CsVecView<N>, b: &[N]) -> N {
    a.iter().nnz_zip(b.iter().cloned().enumerate())
        .map(|(_, a, b)| a*b)
        .fold(N::zero(), |sum, x| sum + x)
}

fn min<N: PartialOrd + Copy, I: Iterator<Item=N>>(mut itr: I) -> Option<N> {
    let mut r = itr.next();
    for v in itr {
        r = r.map(|r| if r < v { r } else { v }).or(Some(v));
    }
    r
}

fn max<N: PartialOrd + Copy, I: Iterator<Item=N>>(mut itr: I) -> Option<N> {
    let mut r = itr.next();
    for v in itr {
        r = r.map(|r| if r > v { r } else { v }).or(Some(v));
    }
    r
}

fn as_cs_vec<'a, N: Copy>(v: &'a [N], indices: &'a [usize]) -> CsVecView<'a, N> {
    assert_eq!(v.len(), indices.len());
    CsVec::new_borrowed(v.len(), indices, v).unwrap()
}

pub trait Eps {
    fn eps() -> Self;
}

impl Eps for f32 {
    fn eps() -> f32 { 1.1920929e-07 }
}

impl Eps for f64 {
    fn eps() -> f64 { 2.220446049250313e-16 }
}

pub fn spsolqp<N: Float + Eps + Default + Debug + rand::Rand + Debug>(q: CsMatView<N>, c: CsVecView<N>, a: CsMatView<N>, b: CsVecView<N>, tolerance: N, step_size: N) -> Result<(Vec<N>, N), SolverError>{
    let (m, n) = (a.rows(), a.cols());
    assert_eq!(q.rows(), n);
    assert_eq!(q.cols(), n);
    assert_eq!(b.dim(), m);

    println!("Searching for a feasible point...");

    let mut alpha = N::from(0.95).unwrap();

    let a_ = &b - &(&a * &from_vec(vec![N::one(); n]));
    //let mut x = from_vec(vec![N::one(); n + 1]);
    let mut x = vec![N::one(); n];
    let n_ind: Vec<_> = (0..n).collect();
    let m_ind: Vec<_> = (0..m).collect();
    let mut z = N::zero();
    let mut ob = N::one();
    let mut obhis = vec![ob];
    let mut gap = ob - z;
    let mut u = vec![];
    while gap >= tolerance {
        let dx = per_element(&x, |x| N::one() / x);
        let dd = CsMat::new_owned(CompressedStorage::CSC, n, n,
                                  (0..n+1).collect(),
                                  (0..n).collect(),
                                  dx.clone()).unwrap();
        let mat = sprs::sparse::construct::bmat(
            &[[Some(dd.borrowed()), Some(a.transpose_view())],
              [Some(a.borrowed()), None]]).unwrap();
        let solver = LdlNumeric::new(&mat);
        let dx_extended = {
            let mut v = dx;
            v.extend(::std::iter::repeat(N::zero()).take(m));
            v
        };
        let a_extended = {
            let mut v = vec![N::zero(); n + m];
            a_.scatter(&mut v[n..]);
            v
        };
        let y1 = solver.solve(&dx_extended);
        let y1 = &y1[n..];
        let y2 = solver.solve(&a_extended);
        let y2 = &y2[n..];
        let r_ob_2 = N::one() / (ob * ob);
        let w1 = (N::one() / ob - dot(a_.borrowed(), y1)) / (r_ob_2 - dot(a_.borrowed(), y2));
        let w2 = N::one() / (r_ob_2 - dot(a_.borrowed(), y2));
        let mut y1: Vec<_> = combine_vec_2(y1, y2, |y1, y2| y1 - w1 * y2);
        let mut y2: Vec<_> = per_element(y2, |y2| -w2 * y2);
        let w1 = dot(b.borrowed(), &y1);
        let w2 = dot(b.borrowed(), &y2);
        y1 = per_element(&y1, |y1| y1 / (N::one() + w1));
        y2 = combine_vec_2(&y1, &y2, |y1, y2| y2 - w2*y1);
        u = combine_vec_2(&(&a.transpose_view() * &as_cs_vec(&y2, &m_ind)), &x, |v, x| -x * v);
        u.push(ob * (N::one() - dot(a_.borrowed(), &y2)));
        u.push(w2 / (N::one() + w1));
        let mut v = combine_vec_2(&(&a.transpose_view() * &as_cs_vec(&y1, &m_ind)), &x, |v, x| x * v);
        v.push(ob * dot(a_.borrowed(), &y1));
        v.push(N::one() / (N::one() + w1));
        if min(combine_vec_2(&u, &v, |u, v| u - z*v).into_iter()).unwrap() >= N::zero() {
            z = dot(b.borrowed(), &combine_vec_2(&y1, &y2, |y1, y2| y2 + z*y1));
        }
        u = combine_vec_2(&u, &v, |u, v| u - z * v - (ob - z)/(N::from(n + 2).unwrap()));
        let nora = max(u.iter().cloned()).unwrap();
        if nora == u.at(n) {
            alpha = N::one();
        }
        v = per_element(&u, |u| N::one() - (alpha/nora)*u);
        x = combine_vec_2(&x, &crop(&v[..], 0, 0, n, n), |x, v_| x * v_ / v.at(n + 1));
        ob = ob * v.at(n) / v.at(n + 1);
        obhis.push(ob);
        gap = ob - z;
        if z > N::zero() {
            return Result::Err(SolverError::NoFeasibleSolution)
        }
    }
    println!("Searching for an optimal solution...");
    alpha = N::from(0.9).unwrap();
    let mut comp: Vec<N> = rand_vec(n);
    let mat = sprs::sparse::construct::bmat(
        &[[Some(CsMat::eye(CompressedStorage::CSC, n).borrowed()), Some(a.transpose_view())],
          [Some(a.borrowed()), None]]).unwrap();
    let solver = LdlNumeric::new(&mat);
    let comp_extended = {
        let mut v = comp;
        v.extend(::std::iter::repeat(N::zero()).take(m));
        v
    };
    comp = solver.solve(&comp_extended);
    comp.truncate(n);
    let comp_div_x = combine_vec_2(&comp, &x, |comp, x| comp / x);
    let mut nora = min(comp_div_x.iter().cloned()).unwrap();
    if nora < N::zero() {
        nora = N::from(-0.01).unwrap() / nora;
    } else {
        nora = max(comp_div_x.iter().cloned()).unwrap();
        if nora == N::zero() {
            println!("The problem has a unique feasible point");
            return Ok((x, ob));
        }
        nora = N::from(0.01).unwrap() / nora;
    }
    x = combine_vec_2(&x, &comp, |x, comp| x + nora * comp);
    let mut obvalue = dot((&q * &as_cs_vec(&x, &n_ind)).borrowed(), &x) * N::from(0.5).unwrap() + dot(c.borrowed(), &x);
    obhis.push(obvalue);
    let mut lower = N::neg_infinity();
    let mut y;
    let mut zhis = vec![lower];
    gap = N::one();
    let mut lambda = N::one().max((obvalue.abs() / N::from(n).unwrap().sqrt().sqrt()));
    let mut iter = 0;
    while gap >= tolerance {
        iter += 1;
        lambda = (N::one() - step_size)*lambda;
        let mut go = N::zero();
        let gg = combine_vec_2(&(&q * &as_cs_vec(&x, &n_ind)), &c, |v, c| v + c);
        let mut xx = CsMat::new_owned(CompressedStorage::CSC, n, n,
                                  (0..n+1).collect(),
                                  (0..n).collect(),
                                  x.clone()).unwrap();
        let aa = &a*&xx;
        xx = &(&xx*&q)*&xx;
        while go <= N::zero() {
            let solver = LdlNumeric::new(&sprs::sparse::construct::bmat(&[
                [Some((&xx + &(&CsMat::eye(CompressedStorage::CSC, n) * lambda)).borrowed()), Some(aa.transpose_view())],
                [Some(aa.borrowed()), None]]).unwrap());
            let mut rhs = combine_vec_2(&x, &gg, |x, gg| -x*gg);
            rhs.extend(::std::iter::repeat(N::zero()).take(m));
            u = solver.solve(&rhs);
            let xx = combine_vec_2(&x, &crop(&u[..], 0, 0, n, n), |x, u| x + x*u);
            go = min(xx.iter().cloned()).unwrap();
            if go > N::zero() {
                let xx = as_cs_vec(&xx, &n_ind);
                ob = N::from(0.5).unwrap() * xx.dot(&(&q * &xx)) + c.dot(&xx);
                go = go.min(obvalue - ob + N::eps());
            }
            lambda = N::from(2.0).unwrap() * lambda;
            if lambda >= (N::one() + obvalue.abs()/tolerance) {
                println!("The problem seems unbounded.");
                return Err(SolverError::Unbounded);
            }
        }
        y = per_element(&crop(&u[..], n, 0, m, m), |u| -u);
        u = per_element(&crop(&u[..], 0, 0, n, n), |u| u);
        nora = min(u.iter().cloned()).unwrap();
        if nora <= N::zero() {
            nora = alpha / (-nora).max(N::one());
        } else {
            nora = N::infinity();
        }
        u = combine_vec_2(&x, &u, |x, u| x*u);
        let w1;
        let w2;
        {
            let u = as_cs_vec(&u, &n_ind);
            w1 = u.dot(&(&q*&u));
            w2 = -dot(u.borrowed(), &gg);
        }
        if w1 > N::zero() {
            nora = (w2/w1).min(nora);
        }
        if nora == N::infinity() {
            ob = N::neg_infinity();
        } else {
            x = combine_vec_2(&x, &u, |x, u| x + nora*u);
            let x = as_cs_vec(&x, &n_ind);
            ob = N::from(0.5).unwrap() * x.dot(&(&q*&x)) + x.dot(&c);
        }
        //END phase2
        if ob == N::neg_infinity() {
            println!("The problem is unbounded");
            return Err(SolverError::Unbounded);
        } else {
            obhis.push(ob);
            comp = vec![N::zero(); n];
            let x = as_cs_vec(&x, &n_ind);
            (&(&(&q * &x) + &c) - &(&a.transpose_view() * &as_cs_vec(&y, &m_ind))).scatter(&mut comp[..]);
            if min(comp.iter().cloned()).unwrap() >= N::zero() {
                zhis.push(ob - dot(x, &comp));
                lower = zhis[iter];
                gap = (ob-lower)/(N::one() + ob.abs());
                obvalue = ob;
            } else {
                let z_new = zhis[iter-1];
                zhis.push(z_new);
                lower = zhis[iter];
                gap = (obvalue - ob)/(N::one() + ob.abs());
                obvalue = ob;
            }
        }
    }
    println!("A (local) optimal solution has been found in {} iterations.", iter);
    Ok((x, ob))
}

#[test]
fn it_works() {
    let q = CsMat::new_owned(CompressedStorage::CSR, 5, 5,
                             vec![0, 2, 4, 4, 4, 4],
                             vec![0, 1,
                                  0, 1],
                             vec![1.0, -1.0,
                                  -1.0, 2.0]).unwrap();
    let c = CsVec::new_owned(5, vec![0, 1], vec![-2.0, -6.0]).unwrap();
    let a = CsMat::new_owned(CompressedStorage::CSR, 3, 5,
                             vec![0, 3, 6, 9],
                             vec![0, 1, 2, 0, 1, 3, 0, 1, 4],
                             vec![ 1.0, 1.0, 1.0,
                                  -1.0, 2.0,      1.0,
                                   2.0, 1.0,           1.0]).unwrap();
    let b = CsVec::new_owned(3, vec![0, 1, 2], vec![2.0, 2.0, 3.0]).unwrap();
    let target_error = 1E-5;
    let (x, ob) = spsolqp::<f64>(
        q.borrowed(), c.borrowed(),
        a.borrowed(), b.borrowed(),
        target_error, 0.95).unwrap();
    let answer = -8.0 - 2.0/9.0;
    let error = ((ob - answer)/answer).abs();
    println!("x: {:#?}", x);
    println!("result: {:#?}", ob);
    println!("true result: {:#?}", answer);
    println!("error: {:?}", error);
    assert!(error <= target_error);
}
