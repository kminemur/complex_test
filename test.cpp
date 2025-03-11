// #include <cassert>
// #include <cstdio>
// #include <cstdlib>
// #include <iostream>
// #include <cmath>
// i#ide <algorithm>
// #include <unistd.h>
// #include <stdlib.h>

#include <sycl/sycl.hpp>
#include <complex>
#include <fstream>

using namespace std; 
using namespace sycl;




typedef struct {
  long int parent;
  long int xyz[3];
  long int atomStart,  atomEnd, nAtoms;
  long int childStart, childEnd, nChildren;
  long int nearStart,  nearEnd, nNear;
  long int farStart,   farEnd,  nFar;
  long int proc;
  std::complex<double> regExpansion[231];
  std::complex<double> irrExpansion[231];
} cube;

// M2L transformation: Convert regular multipole expansion defined about origin A
// into irregular multipole expansion defined about origin B
void reg2irr_translate( 
                          cube* box,
                          long int iBox,
                          int* farNeighbors,
                          double sideLength,
                          int nL,                          // the largest value of L in the expansion
                          int* trL,                         // indices L for triangular array elements
                          int nSzTri,                      // number of elements in triangular array
                          std::complex<double>* cExpansion  // the resulting expansion; accumulate the result
                        ) {
  std::complex<double>  csum;
  int                   L, LL, m, j, k, nSq;
  double                phase[] = {1.0, -1.0};
  double                x, y, z;                            // particle coordinates

  nSq = 2 * nL + 1;

  buffer <int, 1>                         trL_buf{ trL,        range<1>(nSzTri+1) };
  buffer <std::complex<double>, 1> cExpansion_buf{ cExpansion, range<1>(nSzTri+1) };
  buffer               <double, 1>      phase_buf( phase,      range<1>{2});

  // create aa and bb and initialize them on the device
  buffer <std::complex<double>, 1> aa_buf { range<1>((nL+1) * nSq) };  // helper expansion
  buffer <std::complex<double>, 1> bb_buf { range<1>((nL+1) * nSq) };  // expansion to be shifted

  // instruct the runtime by means of referring to nullptr not to copy the buffers back
  aa_buf.set_final_data(nullptr);
  bb_buf.set_final_data(nullptr);
  trL_buf.set_final_data(nullptr);
  phase_buf.set_final_data(nullptr);

  queue q;
  //for(long int aBox = box[iBox].farStart; aBox <= box[iBox].farStart + box[iBox].nFar - 1; aBox++) {

    long int aBox = box[iBox].farStart;
    int jBox = farNeighbors[aBox-1];  // unit-based fortran array

    buffer <std::complex<double>, 1> bExpansion_buf{ box[jBox].regExpansion, range<1>(nSzTri+1) };
    bExpansion_buf.set_final_data(nullptr);

    x = double(box[jBox].xyz[0] - box[iBox].xyz[0]) * sideLength;
    y = double(box[jBox].xyz[1] - box[iBox].xyz[1]) * sideLength;
    z = double(box[jBox].xyz[2] - box[iBox].xyz[2]) * sideLength;

    // compute helper expansion aa
    q.submit([&](handler &cgh) {
      // auto aa = aa_buf.get_access<access::mode::discard_write>(cgh);
      // auto bb = bb_buf.get_access<access::mode::discard_write>(cgh);
      // auto bExpansion_gpu = bExpansion_buf.get_access<access::mode::read>(cgh);
      // auto phase_gpu =           phase_buf.get_access<access::mode::read>(cgh);

       accessor aa(aa_buf, cgh);
       accessor bb(bb_buf, cgh);
       accessor bExpansion_gpu(bExpansion_buf, cgh);
       accessor phase_gpu(phase_buf, cgh);

       cgh.parallel_for<class hexpan>(range<1>(nSq), [nL, nSq, aa, bb, bExpansion_gpu, phase_gpu, x,y,z] (item<1> index) {
        int idx = index[0];                    // worker index
        int Midx = idx - nL;                   // azimuthal index, m
        int Mpos = sycl::abs(Midx);        // positive value of azimuthal index, m
        double ar = 1.0 / sycl::sqrt( sycl::max(x*x + y*y + z*z, 0.0000000001) );
        double ar2 = ar * ar;
        double ar3 = ar * ar2;
        double xar2 =  x * ar2;
        double yar2 = -y * ar2; 
        double inL = double(idx == nL);
        double iMneg = double(Midx < 0);
        double iMpos = double(Midx >= 0);
        double ipm = (idx < nL) - (idx > nL);

        aa[idx]       = inL * ar;                    // (0,0)
        bb[idx]       = inL * bExpansion_gpu[0];

        aa[nSq + idx] = inL * z * ar3                // (1,0) and (1,1)
                      + double(idx != nL) * std::complex<double>(((idx < nL) - (idx > nL)) * x * ar3, -y * ar3); 

        bb[nSq + idx] = inL * bExpansion_gpu[1]
                      - double(Midx == -1) * conj(bExpansion_gpu[2])
                      + double(Midx ==  1) * bExpansion_gpu[2];

        for(int Lidx = 2; Lidx <= nL; Lidx++){
          bb[nSq * Lidx + idx] = double(Mpos <= Lidx) * (iMneg * phase_gpu[Mpos%2] 
                                                         * conj(bExpansion_gpu[Lidx*(Lidx+1)/2+Mpos]) 
                                                         + iMpos * bExpansion_gpu[Lidx*(Lidx+1)/2+Mpos]
                                                        );
          double twoLm1 = double(2*Lidx-1);
          if(Mpos < Lidx) {
            aa[nSq*Lidx + idx] = (z * twoLm1 * aa[nSq*(Lidx-1) + idx] 
                                  - double((Lidx-1)*(Lidx-1) - Midx*Midx) * aa[nSq*(Lidx-2) + idx]
                                 ) * ar2;
            }
          else {
            aa[nSq*Lidx + idx] = twoLm1 * std::complex<double>(ipm * xar2, yar2) * aa[nSq*(Lidx-1) + idx];
            aa[nSq*(Lidx-1) + idx] = 0.0;
          }
        // }
       });
    });

    // q->submit([&](handler &cgh) {
    //   auto cExpansion_gpu = cExpansion_buf.get_access<access::mode::read_write>(cgh);
    //   auto phase_gpu =           phase_buf.get_access<access::mode::read>(cgh);
    //   auto trL_gpu =               trL_buf.get_access<access::mode::read>(cgh);
    //   auto aa =                     aa_buf.get_access<access::mode::read>(cgh);
    //   auto bb =                     bb_buf.get_access<access::mode::read>(cgh);

    //   auto sum = local_accessor<complex<double>, 1>(range<1>(nL+1), cgh);

    //   range<2> global(nSzTri+1, nL+1);
    //   range<2> local(1, nL+1);
    //   cgh.parallel_for<class cc_compute>(nd_range<2>(global,local), [nL, nSq, trL_gpu, phase_gpu, aa, bb, cExpansion_gpu, sum] (nd_item<2> index) {
    //     int i = index.get_global_id(0);
    //     int JmL = index.get_local_id(1);   // local id of a work-item from sub-group (actually, work-group)
    //     int Lidx = trL_gpu[i];
    //     int Midx = i - Lidx * (Lidx + 1)/2;
    //     int Jidx = JmL + Lidx;

    //     sum[JmL] = 0.0;
    //     for(int Kidx = -JmL; Kidx <= JmL; Kidx++) 
    //       sum[JmL] += phase_gpu[(JmL)%2] * bb[nSq*(JmL) + nL+Kidx] * aa[nSq*Jidx + nL+Kidx+Midx];

    //     int offset = 1;
    //     for(int cycle = 0; cycle < sycl::ceil(sycl::log2((double)(nL+1))); cycle++) {
    //       index.barrier(access::fence_space::local_space); // wait for all work-items in the work-group to finish
    //       if((JmL % (2*offset)) == 0 && JmL < nL+1 - offset) {
    //         sum[JmL] += sum[JmL + offset];
    //       }
    //       offset *= 2;
    //     }

    //     if(JmL == 0)
    //       cExpansion_gpu[i] += sum[0];

    //   });
    // });

  //}

  return; 
} 

int main(){
  int nSzTri = 230;
  long int iBox = 73;
  int nL = 20;

  cube box[586];
  int ffarNeighbors[137664];
  std::complex<double> irrExpansion[nSzTri + 1];
  std::complex<double> cc[nSzTri + 1];
  std::fill(cc, cc+nSzTri + 1, std::complex<double>(0., 0.));

  int trL[nSzTri + 1];

  std::ifstream trLFile("fort.109");
  std::ifstream ffarNeighborsFile("fort.103");
  std::ifstream boxFile("fort.99");
  std::ifstream irrExpansionFile("fort.97");

  for (int i = 0; i <= nSzTri; ++i) {
      trLFile >> trL[i];
  }
  for (int i = 0; i < 137664; ++i) {
      ffarNeighborsFile >> ffarNeighbors[i];
  }
  for (int i = 0; i < 586; ++i) {
      boxFile >> box[i].parent >> box[i].xyz[0] >> box[i].xyz[1] >> box[i].xyz[2]
              >> box[i].atomStart >> box[i].atomEnd >> box[i].nAtoms
              >> box[i].childStart >> box[i].childEnd >> box[i].nChildren
              >> box[i].nearStart >> box[i].nearEnd >> box[i].nNear
              >> box[i].farStart >> box[i].farEnd >> box[i].nFar
              >> box[i].proc;
      for (int j = 0; j <= nSzTri; ++j) {
          // boxFile >> box[i].chargeExpansion[j] >> box[i].potentialExpansion[j];
          boxFile >> box[i].regExpansion[j] >> box[i].irrExpansion[j];
      }
  }
  for (int i = 0; i <= nSzTri; ++i) {
      irrExpansionFile >> irrExpansion[i];
  }

  reg2irr_translate(box, iBox, ffarNeighbors, 5.0, nL, trL, nSzTri, cc);


  // void reg2irr_translate( 
  //   cube *box,
  //   long int& iBox,
  //   long int* farNeighbors,
  //   double& sideLength,
  //   int& nL,                          // the largest value of L in the expansion
  //   int* trL,                         // indices L for triangular array elements
  //   int& nSzTri,                      // number of elements in triangular array
  //   std::complex<double>* cExpansion  // the resulting expansion; accumulate the result
}
