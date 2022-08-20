package Main;

import AI.*;

import java.util.Arrays;

public class Main {

    public Main(){
        float[] input = new float[]{1,1,0,0,1};
        Action[] output = new Action[]{Action._1,Action._2,Action._3,Action._4,Action._5};
        int[] layers = new int[]{input.length,10,10,output.length};
        NN nn = new NN(layers, output, ActivationType.ReLU, .001f);

        long startTime = System.currentTimeMillis();
        for(int i = 0; i < 10_000;i++){
            input = new float[5];
            int index = (int)(Math.random()*5);
            input[index] = 1;

            nn.propagation(input);
            nn.backPropagation(output[4-index]);
        }
        System.out.println(Arrays.toString(nn.propagation(input)));
        System.out.println(nn.getCurrentAccuracy());
        long endTime = System.currentTimeMillis();
        System.out.println("Finished: " + (endTime - startTime) + "ms");

    }
}
