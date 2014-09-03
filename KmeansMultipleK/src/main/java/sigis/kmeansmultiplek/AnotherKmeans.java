/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package sigis.kmeansmultiplek;

/**
 *
 * @author asabater
 */
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import static org.apache.commons.math3.util.Precision.round;
 
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
 
public class AnotherKmeans {
 
    // ---- Static
 
    private static final Logger LOG = LoggerFactory.getLogger(AnotherKmeans.class);
    
    String idPath = "/algunid";
    private final String INPUT_PATH = "/home/asabater/Documents/Development/sgps2249";
    private final String BASE_PATH = "/home/asabater/Documents/Development/PruebaMahout" + idPath;
    private final String POINTS_PATH = BASE_PATH + "/points";
    private final String CLUSTERS_PATH = BASE_PATH + "/clusters";
    private final String OUTPUT_PATH = BASE_PATH + "/output";
    private int numberOfCluster;
    private int kmin = 2;
    private int kmax = 7;
    
    public AnotherKmeans(int min, int max, String id){
        this.kmin = min;
        this.kmax = max;
        this.idPath = id;
        
        }
    
    public AnotherKmeans(){
        
    }
 
    public static void main(final String[] args) {
        final AnotherKmeans application = new AnotherKmeans();
 
        try {
            application.start();
        }
        catch (final Exception e) {
            LOG.error("AnotherKmeans failed", e);
        }
    }
 
    // ---- Fields
 
    
    /*private final double[][] points =
        { { 1, 1 }, { 2, 1 }, { 1, 2 },
        { 2, 2 }, { 3, 3 }, { 8, 8 },
        { 9, 8 }, { 8, 9 }, { 9, 9 } };
    */
 
    // ---- Methods
 
    private void start()
        throws Exception {
 
        final Configuration configuration = new Configuration();
        
        this.numberOfCluster = this.kmin;
        // Create input directories for data
        final File pointsDir = new File(POINTS_PATH);
        if (!pointsDir.exists()) {
            pointsDir.mkdir();
        }
        // read the point values and generate vectors from input data
 
        // Write data to sequence hadoop sequence files
        List<DenseVector> vectors = toDenseVector(configuration);
 
        // Write initial centers for clusters
        writeClusterInitialCenters(configuration, vectors);
 
        // Run K-means algorithm
        final Path inputPath = new Path(POINTS_PATH);
        final Path clustersPath = new Path(CLUSTERS_PATH);
        final Path outputPath = new Path(OUTPUT_PATH);
        HadoopUtil.delete(configuration, outputPath);
 
        KMeansDriver.run(configuration, inputPath, clustersPath, outputPath, 0.001, 10, true, 0, false);
 
        // Read and print output values
        readAndPrintOutputValues(configuration);
    }

    
    private void writeClusterInitialCenters(final Configuration conf, List<DenseVector> points)
        throws IOException {
        final Path writerPath = new Path(CLUSTERS_PATH + "/part-00000");
 
        final SequenceFile.Writer writer =
            SequenceFile.createWriter(
                conf,
                SequenceFile.Writer.file(writerPath),
                SequenceFile.Writer.keyClass(Text.class),
                SequenceFile.Writer.valueClass(Kluster.class));
        
        Random rand = new Random();
        for (int i = 0; i < numberOfCluster; i++) {
            int randomNum = rand.nextInt((points.size() - 0) + 1);
            final Vector vec = points.get(randomNum);
 
            // write the initial centers
            final Kluster cluster = new Kluster(vec, i, new EuclideanDistanceMeasure());
            writer.append(new Text(cluster.getIdentifier()), cluster);
        }
 
        writer.close();
    }
 
    private void readAndPrintOutputValues(final Configuration configuration)
        throws IOException {
        final Path input = new Path(OUTPUT_PATH + "/" + Cluster.CLUSTERED_POINTS_DIR + "/part-m-00000");
        //final Path input = new Path(OUTPUT_PATH + "/" + Cluster.FINAL_ITERATION_SUFFIX + "/part-r-00000");
        System.out.println(Cluster.FINAL_ITERATION_SUFFIX);
        System.out.println(Cluster.CLUSTERED_POINTS_DIR);
        
        final SequenceFile.Reader reader =
            new SequenceFile.Reader(
                configuration,
                SequenceFile.Reader.file(input));
 
        final IntWritable key = new IntWritable();
        final WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
 
        while (reader.next(key, value)) {
            LOG.info("{} belongs to cluster {}", value.toString(), key.toString());
            System.out.println(value.toString().substring(18, 28));
            LOG.info("belongs to cluster {}", key.toString());

        }
        reader.close();
    }
 
    
    private List<DenseVector> toDenseVector(Configuration conf) throws FileNotFoundException, IOException{
        List<DenseVector> positions = new ArrayList<DenseVector>();
        DenseVector position;
        BufferedReader br;
        br = new BufferedReader(new FileReader(this.INPUT_PATH));
        
        String sCurrentLine;
		while ((sCurrentLine = br.readLine()) != null) {
			double[] features = new double[3];
                        String[] values = sCurrentLine.split(",");
			for(int indx=0; indx<features.length;indx++){
                                    features[indx] = Float.parseFloat(values[indx+2]);
                                    if (indx == 2){
                                        features[indx] = round(Float.parseFloat(values[indx+3]),2);
                                    }
			}
                 position = new DenseVector(features);
		 positions.add(position);
                }
                
        final Path path = new Path(POINTS_PATH + "/pointsFile");
        FileSystem fs = FileSystem.get(conf);
        SequenceFile.Writer writer = new SequenceFile.Writer(fs,  conf, path, Text.class, VectorWritable.class);
        
        VectorWritable vec = new VectorWritable();
        Integer count = 0;
        
        for(DenseVector vector : positions){
	vec.set(vector);
	writer.append(new Text(count.toString()), vec);
	count++;
        }
	writer.close();
        return positions;
    }
    
}
