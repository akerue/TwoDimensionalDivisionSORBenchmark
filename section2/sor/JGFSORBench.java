/**************************************************************************
*                                                                         *
*             Java Grande Forum Benchmark Suite - MPJ Version 1.0         *
*                                                                         *
*                            produced by                                  *
*                                                                         *
*                  Java Grande Benchmarking Project                       *
*                                                                         *
*                                at                                       *
*                                                                         *
*                Edinburgh Parallel Computing Centre                      *
*                                                                         * 
*                email: epcc-javagrande@epcc.ed.ac.uk                     *
*                                                                         *
*                                                                         *
*      This version copyright (c) The University of Edinburgh, 2001.      *
*                         All rights reserved.                            *
*                                                                         *
**************************************************************************/


package sor;
import com.sun.org.apache.xalan.internal.xsltc.dom.ArrayNodeListIterator;
import jgfutil.*;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Logger;
import mpi.*;

public class JGFSORBench extends SOR implements JGFSection2{ 

  public static int nprocess;
  public static int rank;
  public static int P = 1, Q = 1;
  private int size; 
  private int datasizes[]={1000,1500,2000};
  private static final int JACOBI_NUM_ITER = 100;
  private static final long RANDOM_SEED = 10101010;

  public static int p_row;
  public static int q_row;
  public static int ref_p_row;
  public static int ref_q_row;
  public static int rem_p_row;
  public static int rem_q_row;

  double [][] p_G = null;
  int m_size,n_size,m_length, n_length;

  Random R = new Random(RANDOM_SEED);

  public JGFSORBench(int nprocess, int rank) {
        this.nprocess=nprocess;
        this.rank=rank;
        int[] factors = factorization(nprocess);
        if (factors.length != 1) {
            for (int i = 0; i < factors.length / 2; i++) {
                this.P *= factors[i];
            }
            for (int i = factors.length / 2; i < factors.length; i++) {
                this.Q *= factors[i];
            }
        }
  }

  public void JGFsetsize(int size){
    this.size = size;
  }

  public void JGFinitialise(){

  }
 
  public void JGFkernel() throws MPIException{

  int iup = 0;
  int jup = 0;
  final Logger logger = Logger.getLogger("SORBench");

/* create the array G on process 0 */

  if(rank==0) {
    m_size = datasizes[size];
    n_size = datasizes[size];
  } else {
    m_size = 0;
    n_size = 0;
  }

  double G[][] = RandomMatrix(m_size, n_size,R);

/* create the sub arrays of G */

  p_row = (((datasizes[size] / 2) + P -1) / P)*2;
  q_row = (((datasizes[size] / 2) + Q -1) / Q)*2;
  ref_p_row = p_row;
  rem_p_row = p_row - ((p_row*P) - datasizes[size]);
  ref_q_row = q_row;
  rem_q_row = q_row - ((q_row*Q) - datasizes[size]);
  if(rank/Q == (P-1)){
    if((p_row*P) > datasizes[size]) {
       p_row = rem_p_row;
    }
  }
  if(rank%Q == (Q-1)){
      if((q_row*Q) > datasizes[size]) {
          q_row = rem_q_row;
      }
  }

  p_G = new double [p_row+2][q_row+2];

/* copy or send the values of G to the sub arrays p_G */
   if(rank==0) {

      if(P==1) {
        iup = p_row+1;
      } else {
        iup = p_row+2;
      }
      if(Q==1) {
        jup = q_row+1;
      } else {
        jup = q_row+2;
      }

      // rank 0 のコピー
      for(int i=1;i<iup;i++){
        for(int j=1;j<jup;j++){
          p_G[i][j] = G[i-1][j-1];
        }
      }

      // 端っこのパディング
      for(int i=0;i<iup;i++){
        p_G[i][0] = 0.0;
      }
      for(int j=0;j<jup;j++){
        p_G[0][j] = 0.0; 
      }

      for(int k=1;k<nprocess;k++){
        if(k/Q == P-1) {
            m_length = rem_p_row + 1;
        } else if(k/Q == 0) {
            m_length = p_row + 1;
        } else {
            m_length = p_row + 2;
        }
        if(k%Q == Q-1) {
            n_length = rem_q_row + 1;
        } else if (k%Q == 0) {
            n_length = q_row + 1;
        } else {
            n_length = q_row + 2;
        }

        int m_start, n_start;
        for (int i = 0; i < m_length; i++) {
            m_start = k / Q * p_row - 1;
            if (m_start < 0) {
                m_start = 0;
            }
            n_start = k % Q * q_row - 1;
            if (n_start < 0) {
                n_start = 0;
            }

            MPI.COMM_WORLD.Send(G[m_start], n_start, n_length, MPI.DOUBLE, k, k);
        }
      }

   } else {
       for (int i = 1; i < p_row + 2; i++) {
           MPI.COMM_WORLD.Recv(p_G[i], 0, q_row + 2, MPI.DOUBLE, 0, rank);
       }
   }


   if (rank%P == P-1) {
       for (int i = 0; i < q_row; i++) {
           p_G[p_G.length - 1][i] = 0.0;
       }
   }
   if (rank/Q == Q-1) {
      for (int j = 0; j < p_row; j++) {
         p_G[j][p_G[0].length - 1] = 0.0;
      }
   }

   MPI.COMM_WORLD.Barrier();

    System.gc();
    SORrun(1.25, p_G, JACOBI_NUM_ITER,G);
  }

  public void JGFvalidate(){

//    double refval[] = {0.4984199298207158,1.123010681492097,1.9967774998523777};
    double refval[] = {0.498574406322512,1.1234778980135105,1.9954895063582696};

    if(rank==0) {
      double dev = Math.abs(Gtotal - refval[size]);
      if (dev > 1.0e-12 ){
        System.out.println("Validation failed");
        System.out.println("Gtotal = " + Gtotal + "  " + dev + "  " + size);
      }
    }
  }

  public void JGFtidyup(){
   System.gc();
  }  



  public void JGFrun(int size) throws MPIException{

    if(rank==0) {
      JGFInstrumentor.addTimer("Section2:SOR:Kernel", "Iterations",size);
    }

    JGFsetsize(size); 
    JGFinitialise(); 
    JGFkernel(); 
    JGFvalidate(); 
    JGFtidyup(); 

    if(rank==0){    
      JGFInstrumentor.addOpsToTimer("Section2:SOR:Kernel", (double) (JACOBI_NUM_ITER));
      JGFInstrumentor.printTimer("Section2:SOR:Kernel"); 
    }
  }

 private static double[][] RandomMatrix(int M, int N, java.util.Random R)
  {
                double A[][] = new double[M][N];

        for (int i=0; i<N; i++)
                        for (int j=0; j<N; j++)
                        {
                A[i][j] = R.nextDouble() * 1e-6;
                        }      
                return A;
        }

  private int[] factorization(int num)
  {
      List<Integer> factors = new ArrayList<Integer>();
      while (num % 2 == 0) {
          factors.add(2);
          num /= 2;
      }

      for (int i=3; i * i <= num; i += 2) {
          while(num % i == 0) {
              factors.add(i);
              num /= i;
          }
      }
      if (num > 1) {
          factors.add(num);
      }
      int[] ret = new int[factors.size()];
      for (int i = 0; i < factors.size(); i++) {
          ret[i] = factors.get(i).intValue();
      }
      return ret;
  }
}
