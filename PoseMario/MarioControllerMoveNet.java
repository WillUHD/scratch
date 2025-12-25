import ai.onnxruntime.*;
import nu.pattern.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.videoio.*;

import javax.imageio.*;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import java.io.*;
import java.nio.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

static final String modelPath = "src/movenet-lightning.onnx";
// static final String modelPath = "src/movenet-thunder.onnx";
static final int inDimension = 192, cam = 0;

static final float  backSprintEnd   = -2.0f,
					backSprintStart = -1.1f,
					backStart       = -0.5f,
					runStart        = 0.5f,
					runSprintStart  = 1.1f,
					runSprintEnd    = 2.0f,
					jumpEnd         = -2f,
					jumpStart       = -1.0f,
					zoneLimit       = 0.8f,
					confThresh      = 0f,
					jitterZone      = 20.0f,
					smoothAlpha     = 0.8f;

static final Color  runColor        = new Color(0, 255, 120),
					sprintColor     = new Color(200, 100, 255),
					backColor       = new Color(255, 100, 50),
					jumpColor       = new Color(255, 200, 0),
					idleColor       = new Color(40, 40, 40, 150),
					white           = Color.WHITE;

static final BlockingQueue<Mat> frameQueue = new ArrayBlockingQueue<>(1);
static final AtomicReference<OverlayState> uiState = new AtomicReference<>(new OverlayState());
static final AtomicBoolean  isRunning = new AtomicBoolean(true),
							isPaused = new AtomicBoolean(false);

static volatile BufferedImage cameraImage;
static volatile long resumeEndTime = 0;
static GamePanel panel;

void main() throws Exception {
	OpenCV.loadLocally();

	var window = new JFrame("PoseMario — MoveNet — willuhd");
	window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	window.setAlwaysOnTop(true);

	var layeredPane = new JLayeredPane();
	layeredPane.setPreferredSize(new Dimension(1280, 720));

	panel = new GamePanel();
	panel.setBounds(0, 0, 1280, 720);
	layeredPane.add(panel, JLayeredPane.DEFAULT_LAYER);

	var pauseBtn = new JButton("PAUSE (P)");
	pauseBtn.setBounds(1150, 20, 100, 40);
	pauseBtn.setFocusable(false);
	pauseBtn.addActionListener(e -> togglePause());
	layeredPane.add(pauseBtn, JLayeredPane.PALETTE_LAYER);

	window.add(layeredPane);
	window.pack();
	window.setLocationRelativeTo(null);
	window.setVisible(true);

	window.addComponentListener(new ComponentAdapter() {
		public void componentResized(ComponentEvent e) {
			layeredPane.setPreferredSize(window.getContentPane().getSize());
			panel.setBounds(0, 0, window.getContentPane().getWidth(), window.getContentPane().getHeight());
			pauseBtn.setBounds(window.getContentPane().getWidth() - 130, 20, 100, 40);
		}
	});

	cameraImage = new BufferedImage(1280, 720, BufferedImage.TYPE_3BYTE_BGR);

	var camera = new Thread(new CameraWorker());
	camera.setPriority(Thread.MAX_PRIORITY);
	camera.start();

	var backend = new Thread(new LogicWorker());
	backend.setPriority(Thread.NORM_PRIORITY);
	backend.start();
}

static void togglePause() {
	if (resumeEndTime > 0) return;
	if (isPaused.get()) resumeEndTime = System.currentTimeMillis() + 3000;
	else isPaused.set(true);
}

static class CameraWorker implements Runnable {
	byte[] pixelBuffer;

	public void run() {
		var cap = new VideoCapture(cam);
		cap.set(Videoio.CAP_PROP_BUFFERSIZE, 1);
		cap.set(Videoio.CAP_PROP_FRAME_WIDTH, 1280);
		cap.set(Videoio.CAP_PROP_FRAME_HEIGHT, 720);

		var temp = new Mat();
		var flipped = new Mat();

		int w = 1280, h = 720;
		pixelBuffer = new byte[w * h * 3];

		while (isRunning.get()) {
			if (cap.read(temp)) {
				Core.flip(temp, flipped, 1);
				flipped.get(0, 0, pixelBuffer);
				var dest = ((DataBufferByte) cameraImage.getRaster().getDataBuffer()).getData();
				System.arraycopy(pixelBuffer, 0, dest, 0, pixelBuffer.length);
				panel.repaint();

				if (frameQueue.isEmpty()) {
					frameQueue.offer(flipped.clone());
				}
			}
		}
		cap.release();
	}
}

static class LogicWorker implements Runnable {
	Robot robot;
	Set<Integer> keys = new HashSet<>();
	Set<Integer> target = new HashSet<>();

	int jumpPhase = 0;
	long jumpPhaseStart = 0;
	long lastFpsTime = System.currentTimeMillis();
	int frameCount = 0;
	float currentFps = 0;

	float sX, sY, sScale;
	boolean isTracking = false;

	private final Mat resized, intMat;
	private final int[] pixels;
	private final IntBuffer inBuf;

	LogicWorker() throws AWTException {
		robot = new Robot();
		robot.setAutoDelay(0);
		resized = new Mat();
		intMat = new Mat();
		pixels = new int[inDimension * inDimension * 3];
		inBuf = IntBuffer.allocate(inDimension * inDimension * 3);
	}

	public void run() {
		try (var env = OrtEnvironment.getEnvironment();
			 var opts = new OrtSession.SessionOptions()) {

			opts.addCoreML();
			opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

			try (var session = env.createSession(modelPath, opts)) {
				var inputName = session.getInputNames().iterator().next();
				long[] shape = {1, inDimension, inDimension, 3};

				while (isRunning.get()) {
					Mat raw = null;
					try {
						raw = frameQueue.take();

						long now = System.currentTimeMillis();
						if (resumeEndTime > 0 && now >= resumeEndTime) {
							isPaused.set(false);
							resumeEndTime = 0;
						}

						if (isPaused.get() || resumeEndTime > 0) {
							releaseAllKeys();
							var s = uiState.get();
							s.isResuming = (resumeEndTime > 0);
							s.resumeCount = (int) Math.ceil((resumeEndTime - now) / 1000.0);
							s.fps = 0;
							Thread.sleep(16);
							continue;
						}

						frameCount++;
						if (now - lastFpsTime >= 1000) {
							currentFps = frameCount * 1000.0f / (now - lastFpsTime);
							frameCount = 0;
							lastFpsTime = now;
						}

						Imgproc.resize(raw, resized, new Size(inDimension, inDimension));
						Imgproc.cvtColor(resized, resized, Imgproc.COLOR_BGR2RGB);
						resized.convertTo(intMat, CvType.CV_32S);
						intMat.get(0, 0, pixels);
						inBuf.rewind(); inBuf.put(pixels); inBuf.rewind();

						try (var t = OnnxTensor.createTensor(env, inBuf, shape);
							 var r = session.run(Map.of(inputName, t))) {

							var out = (float[][][][]) r.get(0).getValue();
							var kps = out[0][0];

							var newState = processLogic(kps, raw.cols(), raw.rows(), currentFps);
							uiState.set(newState);
						}
					} catch (InterruptedException e) {
						Thread.currentThread().interrupt();
						break;
					} finally {
						if (raw != null) raw.release();
					}
				}
			}
		} catch (Exception e) { e.printStackTrace(); }
		finally { releaseAllKeys(); }
	}

	OverlayState processLogic(float[][] kps, int w, int h, float fps) {
		var s = new OverlayState();
		s.fps = fps;

		float rawShX = (kps[5][1] + kps[6][1]) * 0.5f;
		float rawShY = (kps[5][0] + kps[6][0]) * 0.5f;
		float rawScale = (float) Math.hypot((kps[5][1] - kps[6][1]) * w, (kps[5][0] - kps[6][0]) * h);

		if (!isTracking) {
			if (rawScale > 30 && Math.abs(rawShX - 0.5f) < 0.15f) {
				isTracking = true;
				sX = rawShX * w;
				sY = rawShY * h;
				sScale = rawScale;
			} else {
				return s;
			}
		} else {
			if (rawScale < 20) {
				isTracking = false;
				releaseAllKeys();
				return s;
			}
		}

		sX = filterValue(sX, rawShX * w);
		sY = filterValue(sY, rawShY * h);
		sScale = filterValue(sScale, rawScale);

		s.midX = sX;
		s.midY = sY;
		s.scale = sScale;
		s.keypoints = kps;

		int actionFlags = 0;
		boolean triggerJump = false;

		int[] wrists = {9, 10};
		int activeWrists = 0;

		for (var v : wrists) {
			if (kps[v][2] < confThresh) continue;
			activeWrists++;

			float nx = ((kps[v][1] * w) - s.midX) / s.scale;
			float ny = ((kps[v][0] * h) - s.midY) / s.scale;

			if (ny > zoneLimit) continue;
			if (ny < jumpStart) triggerJump = true;

			if (nx < backStart) {
				actionFlags |= (nx < backSprintStart) ? 5 : 1;
			} else if (nx > runStart) {
				actionFlags |= (nx < runSprintStart) ? 2 : 4;
			}
		}

		if (activeWrists == 0) return s;

		target.clear();

		if ((actionFlags & 4) != 0) {
			target.add(KeyEvent.VK_R);
			s.activeColor = sprintColor;
			s.label = "SPRINT";
		}

		if ((actionFlags & 1) != 0) {
			target.add(KeyEvent.VK_A);
			if ((actionFlags & 4) == 0) { s.activeColor = backColor; s.label = "BACK"; }
			else s.label = "BACK SPRINT";
		} else if ((actionFlags & 2) != 0 || (actionFlags & 4) != 0) {
			target.add(KeyEvent.VK_D);
			if ((actionFlags & 4) == 0) { s.activeColor = runColor; s.label = "RUN"; }
		} else {
			s.activeColor = idleColor;
			s.label = "NEUTRAL";
		}

		long now = System.currentTimeMillis();
		if (triggerJump) {
			s.label += " + JUMP";
			s.activeColor = jumpColor;

			if (jumpPhase == 0) {
				jumpPhase = 1;
				jumpPhaseStart = now;
			}

			if (jumpPhase == 1) {
				if (now - jumpPhaseStart > 400) {
					jumpPhase = 2;
					jumpPhaseStart = now;
				} else {
					target.add(KeyEvent.VK_SPACE);
				}
			} else if (jumpPhase == 2) {
				if (now - jumpPhaseStart > 75) {
					jumpPhase = 1;
					jumpPhaseStart = now;
					target.add(KeyEvent.VK_SPACE);
				}
			}
		} else {
			jumpPhase = 0;
		}

		for (var k : target) {
			if (!keys.contains(k)) {
				robot.keyPress(k);
				keys.add(k);
			}
		}
		var it = keys.iterator();
		while (it.hasNext()) {
			var k = it.next();
			if (!target.contains(k)) {
				robot.keyRelease(k);
				it.remove();
			}
		}

		return s;
	}

	float filterValue(float current, float target) {
		float diff = target - current;
		if (Math.abs(diff) < jitterZone) return current;
		return current + diff * smoothAlpha;
	}

	void releaseAllKeys() {
		for (var k : keys) robot.keyRelease(k);
		keys.clear();
		jumpPhase = 0;
	}
}

static class OverlayState {
	float[][] keypoints;
	float midX, midY, scale;
	float fps;
	String label = "";
	Color activeColor = null;
	boolean isResuming = false;
	int resumeCount = 0;
}

static class GamePanel extends JPanel {
	BufferedImage watermark;
	int[][] skeleton = {{0,1},{0,2},{1,3},{2,4},{5,6},{5,7},{7,9},{6,8},{8,10},{5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}};

	public GamePanel() {
		try { watermark = ImageIO.read(new File("src/watermark.png")); }
		catch (Exception e) {}
	}

	protected void paintComponent(Graphics g) {
		super.paintComponent(g);
		var g2 = (Graphics2D) g;
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

		int w = getWidth(), h = getHeight();

		if (cameraImage != null) g2.drawImage(cameraImage, 0, 0, w, h, null);
		else { g2.setColor(Color.BLACK); g2.fillRect(0, 0, w, h); return; }

		var s = uiState.get();
		if (s == null) return;

		if (isPaused.get() && !s.isResuming) {
			g2.setColor(new Color(0,0,0, 150));
			g2.fillRect(0,0,w,h);
			drawCenteredText(g2, "Paused", 60, Color.YELLOW, -20);
			drawCenteredText(g2, "", 30, white, 40);
			return;
		}

		if (s.isResuming) {
			g2.setColor(new Color(0,0,0, 100));
			g2.fillRect(0,0,w,h);
			drawCenteredText(g2, "Resuming in", 40, white, -40);
			drawCenteredText(g2, String.valueOf(s.resumeCount), 100, sprintColor, 40);
			return;
		}

		if (s.scale < 10) return;

		double scX = w / 1280.0;
		double scY = h / 720.0;

		boolean isJump = s.label.contains("JUMP");
		String lbl = s.label;

		drawRegion(g2, s, backSprintEnd, runSprintEnd, jumpEnd, jumpStart, "JUMP", jumpColor, isJump, scX, scY);
		drawRegion(g2, s, runStart, runSprintStart, jumpStart, zoneLimit, "RUN", runColor, lbl.contains("RUN"), scX, scY);
		drawRegion(g2, s, runSprintStart, runSprintEnd, jumpStart, zoneLimit, "SPRINT", sprintColor, lbl.contains("SPRINT") && !lbl.contains("BACK"), scX, scY);
		drawRegion(g2, s, backSprintStart, backStart, jumpStart, zoneLimit, "BACK", backColor, lbl.contains("BACK") && !lbl.contains("SPRINT"), scX, scY);
		drawRegion(g2, s, backSprintEnd, backSprintStart, jumpStart, zoneLimit, "SPRINT", sprintColor, lbl.contains("BACK SPRINT"), scX, scY);

		g2.setStroke(new BasicStroke(2));
		g2.setColor(Color.GREEN);
		for (var pair : skeleton) {
			var p1 = s.keypoints[pair[0]];
			var p2 = s.keypoints[pair[1]];
			if (p1[2] > confThresh && p2[2] > confThresh)
				g2.drawLine((int)(p1[1]*w), (int)(p1[0]*h), (int)(p2[1]*w), (int)(p2[0]*h));
		}

		if (watermark != null) {
			var aspect = (double) watermark.getHeight() / watermark.getWidth();
			var wmH = (int) (w * aspect);
			g2.drawImage(watermark, 0, h - wmH, w, wmH, null);
		}

		drawText(g2, String.format("FPS: %.1f", s.fps), 10, 50, runColor);

	}

	void drawRegion(Graphics2D g2, OverlayState s, float rx1, float rx2, float ry1, float ry2,
					String lbl, Color c, boolean active, double sx, double sy) {

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

		g2.setColor(active ? new Color(c.getRed(), c.getGreen(), c.getBlue(), 120) : idleColor);
		g2.fillRect(bx, by, bw, bh);

		if (active) {
			g2.setColor(white);
			g2.setStroke(new BasicStroke(3));
			g2.drawRect(bx, by, bw, bh);
		}

		if (lbl != null && !lbl.isEmpty()) {
			g2.setFont(new Font("SF Pro Display", Font.BOLD, active ? 24 : 12));
			FontMetrics fm = g2.getFontMetrics();
			int tx = bx + (bw - fm.stringWidth(lbl)) / 2;
			int ty = by + (bh - fm.getHeight()) / 2 + fm.getAscent();
			g2.setColor(white);
			g2.drawString(lbl, tx, ty);
		}
	}

	void drawCenteredText(Graphics2D g2, String txt, int size, Color c, int yOffset) {
		g2.setFont(new Font("SF Pro Display", Font.BOLD, size));
		FontMetrics fm = g2.getFontMetrics();
		int x = (getWidth() - fm.stringWidth(txt)) / 2;
		int y = (getHeight() / 2) + (fm.getAscent() / 3) + yOffset;
		g2.setColor(Color.BLACK);
		g2.drawString(txt, x+2, y+2);
		g2.setColor(c);
		g2.drawString(txt, x, y);
	}

	void drawText(Graphics2D g2, String txt, int x, int y, Color c) {
		g2.setFont(new Font("SF Mono", Font.BOLD, 28));
		g2.setColor(Color.BLACK);
		g2.drawString(txt, x+1, y+1);
		g2.setColor(c);
		g2.drawString(txt, x, y);
	}
}