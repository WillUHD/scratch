import ai.onnxruntime.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.videoio.*;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import java.nio.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.atomic.*;

static String   modelPath = "src/yolo11n-pose.onnx";
static int      camId = 0,
                inputSize = 640,
                L_SH = 5,
                R_SH = 6,
                L_WR = 9,
                R_WR = 10;
static float    runIn = 0.5f,
                sprintIn = 1.3f,
                sprintOut = 2.5f,
                backSprintOut = -sprintOut,
                backSprintIn = -sprintIn,
                backIn = -runIn,
                jumpTop = -2.5f,
                jumpBottom = -0.8f,
                neutralTop = -0.8f,
                neutralBottom = 0.8f,
                duckTop = 0.8f,
                duckBottom = 2.5f;
static Color    cRun = new Color(0, 255, 120),
                cSprint = new Color(200, 100, 255),
                cBack = new Color(255, 100, 50),
                cJump = new Color(255, 200, 0),
                cDuck = new Color(255, 200, 0),
                cIdle = new Color(40, 40, 40, 150),
                cWhite = Color.WHITE,
                cSkel = Color.GREEN;

static AtomicReference<BufferedImage> latestImage = new AtomicReference<>();
static AtomicReference<Mat> latestMat = new AtomicReference<>();
static AtomicReference<OverlayState> latestState = new AtomicReference<>(new OverlayState());
static AtomicBoolean isRunning = new AtomicBoolean(true);

static int[] skeletonPairs = {
        0,1, 0,2, 1,3, 2,4, 5,6, 5,7, 7,9, 6,8, 8,10, 5,11, 6,12, 11,12, 11,13, 13,15, 12,14, 14,16
};

record RegionDef(float x1, float x2, float y1, float y2, String label) {}

static List<RegionDef> bgRegions = List.of(
        new RegionDef(backSprintOut, backSprintIn, neutralTop, neutralBottom, "SPRINT"),
        new RegionDef(backSprintIn, backIn, neutralTop, neutralBottom, "BACK"),
        new RegionDef(-0.5f, 0.5f, jumpTop, jumpBottom, "JUMP"),
        new RegionDef(-0.5f, 0.5f, duckTop, duckBottom, "DUCK"),
        new RegionDef(runIn, sprintIn, neutralTop, neutralBottom, "RUN"),
        new RegionDef(sprintIn, sprintOut, neutralTop, neutralBottom, "SPRINT")
);

void main() {
    nu.pattern.OpenCV.loadLocally();

    var window = new JFrame("PoseMario — YOLO11n — willuhd");
    window.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
    window.setAlwaysOnTop(true);

    var panel = new GamePanel();
    window.add(panel);

    window.setPreferredSize(new Dimension(1280, 720));
    window.pack();
    window.setLocationRelativeTo(null);
    window.setVisible(true);

    window.addWindowListener(new WindowAdapter() {
        public void windowClosing(WindowEvent e) {
            isRunning.set(false);
            window.dispose();
            System.exit(0);
        }
    });

    var camThread = new Thread(new CameraWorker(panel));
    camThread.setPriority(Thread.MAX_PRIORITY);
    camThread.start();

    var infThread = new Thread(new InferenceWorker());
    infThread.setPriority(Thread.NORM_PRIORITY);
    infThread.setDaemon(true);
    infThread.start();
}

static class OverlayState {
    boolean hasDetection = false;
    float midX, midY, scale;
    float[] keypoints, scores;
    String activeLabel = "";
    Color activeColor = null;
    RegionDef activeHighlightRect = null; // Calculated dynamically
    String debugText = "";
}

static class CameraWorker implements Runnable {
    GamePanel panel;
    CameraWorker(GamePanel p) { this.panel = p; }

    @Override
    public void run() {
        var cap = new VideoCapture(camId);
        cap.set(Videoio.CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(Videoio.CAP_PROP_FRAME_HEIGHT, 720);

        if (!cap.isOpened()) {
            System.err.println("Error: Camera not found.");
            return;
        }

        var frame = new Mat();
        while (isRunning.get())
            if (cap.read(frame) && !frame.empty()) {
                Core.flip(frame, frame, 1);
                latestMat.set(frame.clone());
                latestImage.set(matToBufferedImage(frame));
                panel.repaint();
            }
        cap.release();
    }

    private BufferedImage matToBufferedImage(Mat m) {
        int w = m.cols(), h = m.rows();
        var img = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);
        var data = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        m.get(0, 0, data);
        return img;
    }
}

static class InferenceWorker implements Runnable {
    @Override
    public void run() {
        try (var env = OrtEnvironment.getEnvironment();
             var opts = new OrtSession.SessionOptions()) {
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            opts.addCoreML();
//            opts.addCPU(true);

            try (var session = env.createSession(modelPath, opts)) {
                var inputName = session.getInputNames().iterator().next();
                var robot = new Robot();
                robot.setAutoDelay(0);

                Set<Integer> currentKeys = new HashSet<>();
                long lastTime = System.nanoTime();
                int frames = 0;
                float fps = 0;

                var resized = new Mat();
                var floatBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
                        .order(ByteOrder.nativeOrder()).asFloatBuffer();
                byte[] pixelData = new byte[inputSize * inputSize * 3];

                while (isRunning.get()) {
                    var source = latestMat.getAndSet(null);
                    if (source == null) {
                        Thread.sleep(5);
                        continue;
                    }

                    int h = source.rows();
                    int w = source.cols();
                    float scaleFactor = Math.min((float)inputSize/h, (float)inputSize/w);
                    int nw = (int)(w * scaleFactor);
                    int nh = (int)(h * scaleFactor);
                    int padX = (inputSize - nw) / 2;
                    int padY = (inputSize - nh) / 2;

                    Imgproc.resize(source, resized, new Size(nw, nh));
                    if (resized.channels() != 3) { source.release(); continue; }
                    resized.get(0, 0, pixelData);
                    floatBuffer.clear();
                    int plane = inputSize * inputSize;
                    for(int i=0; i<floatBuffer.capacity(); i++) floatBuffer.put(0f);

                    int pIdx = 0;
                    for(int y=0; y<nh; y++)
                        for(int x=0; x<nw; x++) {
                            float b = (pixelData[pIdx++] & 0xFF) / 255f;
                            float g = (pixelData[pIdx++] & 0xFF) / 255f;
                            float r = (pixelData[pIdx++] & 0xFF) / 255f;
                            int bufIdx = (y + padY) * inputSize + (x + padX);
                            floatBuffer.put(bufIdx, r);
                            floatBuffer.put(bufIdx + plane, g);
                            floatBuffer.put(bufIdx + plane * 2, b);
                        }
                    floatBuffer.rewind();

                    var newState = new OverlayState();
                    long[] shape = {1, 3, inputSize, inputSize};

                    try (var tensor = OnnxTensor.createTensor(env, floatBuffer, shape);
                         var res = session.run(Map.of(inputName, tensor))) {

                        var output = (float[][][]) res.get(0).getValue();
                        var data = output[0];

                        int bestIdx = -1;
                        float maxScore = -1f;
                        var scoresArr = data[4];
                        for(int i=0; i<scoresArr.length; i++)
                            if (scoresArr[i] > maxScore) {
                                maxScore = scoresArr[i];
                                bestIdx = i;
                            }

                        if (maxScore > 0.5f) {
                            newState.hasDetection = true;
                            newState.keypoints = new float[17 * 2];
                            newState.scores = new float[17];

                            for (int k=0; k<17; k++) {
                                float kx = (data[5 + k*3][bestIdx] - padX) / scaleFactor;
                                float ky = (data[5 + k*3 + 1][bestIdx] - padY) / scaleFactor;
                                float ks = data[5 + k*3 + 2][bestIdx];
                                newState.keypoints[k*2] = kx;
                                newState.keypoints[k*2+1] = ky;
                                newState.scores[k] = ks;
                            }

                            if (newState.scores[L_SH] > 0.5f && newState.scores[R_SH] > 0.5f) {
                                float sx1 = newState.keypoints[L_SH*2], sy1 = newState.keypoints[L_SH*2+1];
                                float sx2 = newState.keypoints[R_SH*2], sy2 = newState.keypoints[R_SH*2+1];

                                newState.midX = (sx1 + sx2) / 2f;
                                newState.midY = (sy1 + sy2) / 2f;
                                newState.scale = (float) Math.hypot(sx2 - sx1, sy2 - sy1);

                                if (newState.scale > 30) {
                                    Set<Integer> targetKeys = new HashSet<>();
                                    var hState = "NEUTRAL";
                                    var vState = "NEUTRAL";

                                    int[] wrists = {L_WR, R_WR};
                                    for(int wid : wrists)
                                        if (newState.scores[wid] > 0.5f) {
                                            float wx = newState.keypoints[wid*2];
                                            float wy = newState.keypoints[wid*2+1];
                                            float nx = (wx - newState.midX) / newState.scale;
                                            float ny = (wy - newState.midY) / newState.scale;

                                            if (nx < backIn) hState = (nx < backSprintIn) ? "BACK_SPRINT" : "BACK";
                                            else if (nx > runIn) hState = (nx < sprintIn) ? "RUN" : "SPRINT";

                                            if (ny < jumpBottom) vState = "JUMP";
                                            else if (ny > duckTop) vState = "DUCK";
                                        }

                                    float rx1 = 0, rx2 = 0, ry1 = neutralTop, ry2 = neutralBottom;

                                    if ("DUCK".equals(vState)) {
                                        newState.activeLabel = "DUCK";
                                        newState.activeColor = cDuck;
                                        rx1 = -0.5f; rx2 = 0.5f; ry1 = duckTop; ry2 = duckBottom;
                                        targetKeys.add(KeyEvent.VK_S);
                                    } else {
                                        switch (hState) {
                                            case "SPRINT" -> {
                                                newState.activeLabel = "SPRINT";
                                                newState.activeColor = cSprint;
                                                rx1 = runIn; rx2 = sprintOut;
                                                targetKeys.add(KeyEvent.VK_D);
                                                targetKeys.add(KeyEvent.VK_R);
                                            }
                                            case "RUN" -> {
                                                newState.activeLabel = "RUN";
                                                newState.activeColor = cRun;
                                                rx1 = runIn; rx2 = sprintIn;
                                                targetKeys.add(KeyEvent.VK_D);
                                            }
                                            case "BACK_SPRINT" -> {
                                                newState.activeLabel = "SPRINT BACK";
                                                newState.activeColor = cSprint;
                                                rx1 = backSprintOut; rx2 = backIn;
                                                targetKeys.add(KeyEvent.VK_A);
                                                targetKeys.add(KeyEvent.VK_R);
                                            }
                                            case "BACK" -> {
                                                newState.activeLabel = "BACK";
                                                newState.activeColor = cBack;
                                                rx1 = backSprintIn; rx2 = backIn;
                                                targetKeys.add(KeyEvent.VK_A);
                                            }
                                        }

                                        if ("JUMP".equals(vState)) {
                                            newState.activeLabel = newState.activeLabel.isEmpty() ? "JUMP" : newState.activeLabel + " + JUMP";
                                            targetKeys.add(KeyEvent.VK_SPACE);
                                            ry1 = jumpTop; ry2 = jumpBottom;
                                            if (newState.activeColor == null) {
                                                newState.activeColor = cJump;
                                                rx1 = -0.5f; rx2 = 0.5f;
                                            }
                                        }
                                    }

                                    if (newState.activeColor != null)
                                        newState.activeHighlightRect = new RegionDef(rx1, rx2, ry1, ry2, "");

                                    for(var k : targetKeys)
                                        if(!currentKeys.contains(k)) {
                                            robot.keyPress(k);
                                            currentKeys.add(k);
                                        }

                                    var it = currentKeys.iterator();
                                    while(it.hasNext()) {
                                        Integer k = it.next();
                                        if(!targetKeys.contains(k)) { robot.keyRelease(k); it.remove(); }
                                    }
                                }
                            }
                        }
                    }

                    frames++;
                    long now = System.nanoTime();
                    if (now - lastTime >= 1e9) {
                        fps = frames / ((now - lastTime) / 1e9f);
                        frames = 0;
                        lastTime = now;
                    }
                    newState.debugText = String.format("overlay: %.1f FPS", fps);

                    latestState.set(newState);
                    source.release();
                }
            }
        } catch (Exception e) { e.printStackTrace(); }
    }
}

static class GamePanel extends JPanel {
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        var g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        var img = latestImage.get();
        var state = latestState.get();
        int w = getWidth();
        int h = getHeight();

        if (img != null)
            g2.drawImage(img, 0, 0, w, h, null);
        else {
            g2.setColor(Color.BLACK);
            g2.fillRect(0,0,w,h);
            return;
        }

        if (state == null || !state.hasDetection || state.scale < 30) {
            drawText(g2, "No Pose / Too Far", 20, 50, Color.RED);
            return;
        }

        double scaleX = w / (double) img.getWidth();
        double scaleY = h / (double) img.getHeight();

        for (var reg : bgRegions)
            drawRegion(g2, state, reg.x1(), reg.x2(), reg.y1(), reg.y2(),
                    reg.label(), cIdle, false, scaleX, scaleY);

        if (state.activeHighlightRect != null && state.activeColor != null) {
            var r = state.activeHighlightRect;
            drawRegion(g2, state, r.x1(), r.x2(), r.y1(), r.y2(),
                    state.activeLabel, state.activeColor, true, scaleX, scaleY);
        }

        g2.setColor(cSkel);
        g2.setStroke(new BasicStroke(2));
        var kp = state.keypoints;
        for (int i=0; i<skeletonPairs.length; i+=2) {
            int i1 = skeletonPairs[i];
            int i2 = skeletonPairs[i+1];
            if (state.scores[i1] > 0.5f && state.scores[i2] > 0.5f) {
                int x1 = (int) (kp[i1*2] * scaleX);
                int y1 = (int) (kp[i1*2+1] * scaleY);
                int x2 = (int) (kp[i2*2] * scaleX);
                int y2 = (int) (kp[i2*2+1] * scaleY);
                g2.drawLine(x1, y1, x2, y2);
            }
        }

        drawText(g2, state.debugText, 10, 20, Color.YELLOW);
    }

    private void drawRegion(Graphics2D g2, OverlayState s, float rx1, float rx2, float ry1, float ry2,
                            String lbl, Color c, boolean highlight, double sx, double sy) {

        float ix1 = s.midX + rx1 * s.scale;
        float iy1 = s.midY + ry1 * s.scale;
        float ix2 = s.midX + rx2 * s.scale;
        float iy2 = s.midY + ry2 * s.scale;

        int x1 = (int) (ix1 * sx);
        int y1 = (int) (iy1 * sy);
        int x2 = (int) (ix2 * sx);
        int y2 = (int) (iy2 * sy);

        int bx = Math.min(x1, x2);
        int by = Math.min(y1, y2);
        int bw = Math.abs(x1 - x2);
        int bh = Math.abs(y1 - y2);

        g2.setColor(highlight ? new Color(c.getRed(), c.getGreen(), c.getBlue(), 120) : c);
        g2.fillRect(bx, by, bw, bh);

        if (highlight) {
            g2.setColor(cWhite);
            g2.setStroke(new BasicStroke(3));
            g2.drawRect(bx, by, bw, bh);
        }

        if (lbl != null && !lbl.isEmpty()) {
            g2.setFont(new Font("Helvetica Neue", Font.BOLD, highlight ? 24 : 12));
            var fm = g2.getFontMetrics();
            int tx = bx + (bw - fm.stringWidth(lbl)) / 2;
            int ty = by + (bh - fm.getHeight()) / 2 + fm.getAscent();
            g2.setColor(cWhite);
            g2.drawString(lbl, tx, ty);
        }
    }

    private void drawText(Graphics2D g2, String txt, int x, int y, Color c) {
        g2.setFont(new Font("Helvetica Neue", Font.BOLD, 14));
        g2.setColor(Color.BLACK);
        g2.drawString(txt, x+1, y+1);
        g2.setColor(c);
        g2.drawString(txt, x, y);
    }
}