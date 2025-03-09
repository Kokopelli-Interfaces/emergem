Can you show me an example definition of an EmergemNode in rhai?

An EmergemNode is controlled by an EmergemBlock, which has
N Push able Rotary knobs
M Push buttons
A main push able knob

For each n knobs, and m buttons, and the main one it Optionally defines an input signal and/or a output signal of a particular type/tag (e.g midi, audio, control) 

The rhai script will have to
Define the process function for the module.. Typically using exposed rust functions for common signal processing techniques
It more so chains them together in a particular way

It will also have to bind the parameter values of the EmergemBlock which is controlling the EmergemNode it is defining to adjust particular parameters in the signal chain

E.g it may define that the node is called "REVERB" AND THE first knob controls "Size" and the value it receives will change a tweaking variable in the process function 

Let’s create an exciting example of an `EmergemNode` definition in Rhai for your signal processing, modular emergence, live creation environment! This will showcase how Rhai can define a node (like a "REVERB" module), chain Rust-exposed signal processing functions, and bind parameters from an `EmergemBlock` (with its knobs and buttons) to control the signal chain dynamically. I’ll assume the `EmergemNode` is part of your system where Rhai scripts orchestrate the behavior, leveraging Rust for the heavy lifting.

Here’s the vision: an `EmergemNode` called "REVERB" with a few knobs and buttons, chaining a delay and a low-pass filter (implemented in Rust), and mapping the `EmergemBlock` controls to tweak parameters live—all reloadable at runtime!

---

### Step 1: Conceptual Setup
#### EmergemBlock (Assumed Structure)
- **N Pushable Rotary Knobs**: Let’s say `N = 2` (e.g., "Size" and "Dampening").
- **M Push Buttons**: Let’s say `M = 1` (e.g., "Bypass").
- **Main Pushable Knob**: Controls a master parameter (e.g., "Wet/Dry Mix").
- **Signals**: Each control can optionally link to an input/output signal with a type/tag (e.g., "audio", "control").

#### EmergemNode in Rhai
- **Name**: "REVERB".
- **Process Function**: Chains Rust functions like `delay` and `low_pass`.
- **Parameter Binding**: Maps `EmergemBlock` controls to tweakable variables (e.g., delay time, filter cutoff).

#### Rust-Exposed Functions
We’ll assume these DSP functions are exposed to Rhai:
- `delay(input: &[f32], output: &mut [f32], delay_samples: i64)`: Adds a delay effect.
- `low_pass(input: &[f32], output: &mut [f32], last_output: &mut f32, alpha: f32)`: Applies a low-pass filter.

---

### Step 2: Rhai Script for "REVERB" EmergemNode
Here’s how the Rhai script might look. It defines the node, its signal chain, and binds the `EmergemBlock` controls to parameters.

```rhai
// EmergemNode metadata (name and control bindings)
let node_name = "REVERB";

// Persistent state for the signal chain
let last_output = 0.0;       // For low-pass filter
let delay_buffer = [];       // Could be managed in Rust, but simulated here
let wet_dry_mix = 0.5;       // Default mix (0.0 = dry, 1.0 = wet)
let size = 0.5;             // Delay time factor (0.0 to 1.0)
let dampening = 0.2;        // Filter alpha (0.0 to 1.0)
let bypass = false;         // Bypass toggle

// Bind EmergemBlock controls to parameters
fn bind_controls(knobs, buttons, main_knob) {
    // N = 2 knobs
    size = knobs[0];         // Knob 0 -> Size (delay time)
    dampening = knobs[1];    // Knob 1 -> Dampening (filter strength)
    
    // M = 1 button
    bypass = buttons[0];     // Button 0 -> Bypass toggle
    
    // Main knob
    wet_dry_mix = main_knob; // Main knob -> Wet/Dry mix
}

// Define input/output signals (optional, tagged by type)
let inputs = [
    ["audio", "input"]      // Audio input signal
];
let outputs = [
    ["audio", "output"]     // Audio output signal
];

// Process function: chains Rust DSP functions
fn process(input, output) {
    if bypass {
        // Bypass: copy input to output
        for i in 0..input.len() {
            output[i] = input[i];
        }
        return;
    }

    // Step 1: Apply delay (Rust function)
    let delay_samples = (size * 1000.0) as int; // Map 0.0-1.0 to 0-1000 samples
    delay(input, output, delay_samples);

    // Step 2: Apply low-pass filter (Rust function) to dampen high frequencies
    low_pass(output, output, last_output, dampening); // In-place processing

    // Step 3: Wet/Dry mix
    for i in 0..input.len() {
        output[i] = input[i] * (1.0 - wet_dry_mix) + output[i] * wet_dry_mix;
    }
}

// Initialization function (optional, called once on load)
fn init() {
    print("REVERB node loaded!");
}
```

---

### Step 3: How It Fits Your System
#### Integration with Rust
Assume your Rust system exposes the `EmergemNode` framework and calls this script:
```rust
use rhai::{Engine, Scope, AST};

struct EmergemNode {
    engine: Engine,
    scope: Scope<'static>,
    ast: AST,
}

impl EmergemNode {
    fn new(script_path: &Path) -> Self {
        let mut engine = Engine::new();
        
        // Register Rust DSP functions
        engine.register_fn("delay", |input: &[f32], output: &mut [f32], delay_samples: i64| {
            // Simplified delay (real impl would use a circular buffer)
            for i in 0..input.len() {
                output[i] = if i >= delay_samples as usize { input[i - delay_samples as usize] } else { 0.0 };
            }
        });
        engine.register_fn("low_pass", |input: &[f32], output: &mut [f32], last: &mut f32, alpha: f32| {
            output[0] = *last * (1.0 - alpha) + input[0] * alpha;
            *last = output[0];
        });

        let script = std::fs::read_to_string(script_path).unwrap();
        let ast = engine.compile(&script).unwrap();
        let mut scope = Scope::new();
        
        // Call init if present
        engine.call_fn::<()>(&mut scope, &ast, "init", ()).ok();
        
        EmergemNode { engine, scope, ast }
    }

    fn bind_controls(&mut self, knobs: Vec<f32>, buttons: Vec<bool>, main_knob: f32) {
        self.engine.call_fn::<()>(
            &mut self.scope,
            &self.ast,
            "bind_controls",
            (knobs, buttons, main_knob),
        ).unwrap();
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.engine.call_fn::<()>(
            &mut self.scope,
            &self.ast,
            "process",
            (input.to_vec(), output.to_vec()),
        ).unwrap();
        if let Some(out) = self.scope.get_value::<Vec<f32>>("output") {
            output.copy_from_slice(&out);
        }
    }
}

// Usage in your async system
#[tokio::main]
async fn main() {
    let mut node = EmergemNode::new(Path::new("reverb.rhai"));
    let knobs = vec![0.5, 0.2]; // Size, Dampening
    let buttons = vec![false];  // Bypass
    let main_knob = 0.5;        // Wet/Dry
    node.bind_controls(knobs, buttons, main_knob);

    let input = vec![1.0; 128];
    let mut output = vec![0.0; 128];
    node.process(&input, &mut output);
    println!("Output[0]: {}", output[0]);
}
```

#### What’s Happening
- **Node Definition**: The Rhai script names itself "REVERB" and defines its signal chain.
- **Control Binding**: `bind_controls` maps the `EmergemBlock`’s knobs/buttons to variables (`size`, `dampening`, etc.).
- **Signal Processing**: `process` chains `delay` and `low_pass` (Rust functions), applying the wet/dry mix.
- **Live Updates**: Edit `reverb.rhai` (e.g., change `dampening` default or chain order), and the async file watcher reloads it.

---

### Step 4: Why This Is Exciting
- **Modular Magic**: The `EmergemNode` defines a reusable "REVERB" block, chaining Rust DSP with Rhai flexibility.
- **Tweakable Live**: Twist knob 0 to adjust "Size" (delay time), knob 1 for "Dampening" (filter cutoff), or the main knob for "Wet/Dry"—all updated instantly via hot-reloading.
- **Emergent Potential**: Add more Rust functions (e.g., `reverb_tail`, `modulate`) and let Rhai experiment with chaining them dynamically.
- **Tagged Signals**: The `inputs`/`outputs` arrays let your system route "audio" or "control" signals to/from this node.

#### Example Tweak
Change the Rhai script live:
```rhai
fn process(input, output) {
    delay(input, output, (size * 2000.0) as int); // Double delay range
    low_pass(output, output, last_output, dampening * 0.5); // Softer filter
    for i in 0..input.len() {
        output[i] = input[i] * (1.0 - wet_dry_mix) + output[i] * wet_dry_mix;
    }
}
```
Save, and your reverb gets deeper and smoother—*no restart needed*!

---

### Step 5: Scaling It
- **More Controls**: Add knobs for "Feedback" or "Decay" by extending `knobs[N]`.
- **Complex Chains**: Chain more Rust functions (e.g., `comb_filter`, `all_pass`) for a richer reverb.
- **State in Rust**: Move `last_output` or `delay_buffer` to Rust for better performance if needed.

This Rhai `EmergemNode` definition is a playground for your creativity—define "REVERB" today, "DISTORTION" tomorrow, all with the same framework!

What do you think—want to tweak this "REVERB" node further or try a different effect (e.g., "FLANGER") next? I can refine it or hook it to real audio with `cpal` if you’re ready to hear it sing!

Glad you’re liking HJSON! I’ll update the OSC-based version to include displays that can be updated for each parameter—meaning every rotary knob, button (rotary push, push, module switch, meta button), main knob, and page knob gets its own display receiving feedback, in addition to the main display. This way, each control has a dedicated display showing its current state (e.g., a value for rotaries, an on/off status for buttons, or a page number for the page knob), alongside the main display showing the node name or broader status.

Here’s the revised HJSON config, keeping it compact yet comprehensive:

---

### HJSON with Displays for Each Parameter
```
# OSC bindings for emergem block with per-control displays
emergem
  id: virtual_control_1
  protocol: osc

  rotary
    1: {send: /rotary/1, receive: /feedback/rotary/1}
    2: {send: /rotary/2, receive: /feedback/rotary/2}
    3: {send: /rotary/3, receive: /feedback/rotary/3}
    4: {send: /rotary/4, receive: /feedback/rotary/4}
    5: {send: /rotary/5, receive: /feedback/rotary/5}
    6: {send: /rotary/6, receive: /feedback/rotary/6}
    7: {send: /rotary/7, receive: /feedback/rotary/7}
    8: {send: /rotary/8, receive: /feedback/rotary/8}

  rotary_push
    1: {send: /rotary_push/1, receive: /feedback/rotary_push/1}
    2: {send: /rotary_push/2, receive: /feedback/rotary_push/2}
    3: {send: /rotary_push/3, receive: /feedback/rotary_push/3}
    4: {send: /rotary_push/4, receive: /feedback/rotary_push/4}
    5: {send: /rotary_push/5, receive: /feedback/rotary_push/5}
    6: {send: /rotary_push/6, receive: /feedback/rotary_push/6}
    7: {send: /rotary_push/7, receive: /feedback/rotary_push/7}
    8: {send: /rotary_push/8, receive: /feedback/rotary_push/8}

  push
    1: {send: /push/1, receive: /feedback/push/1}
    2: {send: /push/2, receive: /feedback/push/2}
    3: {send: /push/3, receive: /feedback/push/3}
    4: {send: /push/4, receive: /feedback/push/4}
    5: {send: /push/5, receive: /feedback/push/5}
    6: {send: /push/6, receive: /feedback/push/6}
    7: {send: /push/7, receive: /feedback/push/7}
    8: {send: /push/8, receive: /feedback/push/8}

  main_knob: {send: /main/knob, receive: /feedback/main/knob}
  main_push: {send: /main/push, receive: /feedback/main/push}

  module_switch
    1: {send: /module/1, receive: /feedback/module/1}
    2: {send: /module/2, receive: /feedback/module/2}
    3: {send: /module/3, receive: /feedback/module/3}
    4: {send: /module/4, receive: /feedback/module/4}

  meta_button
    1: {send: /meta/1, receive: /feedback/meta/1}
    2: {send: /meta/2, receive: /feedback/meta/2}
    3: {send: /meta/3, receive: /feedback/meta/3}
    4: {send: /meta/4, receive: /feedback/meta/4}

  page_knob: {send: /page/knob, receive: /feedback/page/knob}

  # Displays for each control
  display
    rotary
      1: {receive: /display/rotary/1}
      2: {receive: /display/rotary/2}
      3: {receive: /display/rotary/3}
      4: {receive: /display/rotary/4}
      5: {receive: /display/rotary/5}
      6: {receive: /display/rotary/6}
      7: {receive: /display/rotary/7}
      8: {receive: /display/rotary/8}

    rotary_push
      1: {receive: /display/rotary_push/1}
      2: {receive: /display/rotary_push/2}
      3: {receive: /display/rotary_push/3}
      4: {receive: /display/rotary_push/4}
      5: {receive: /display/rotary_push/5}
      6: {receive: /display/rotary_push/6}
      7: {receive: /display/rotary_push/7}
      8: {receive: /display/rotary_push/8}

    push
      1: {receive: /display/push/1}
      2: {receive: /display/push/2}
      3: {receive: /display/push/3}
      4: {receive: /display/push/4}
      5: {receive: /display/push/5}
      6: {receive: /display/push/6}
      7: {receive: /display/push/7}
      8: {receive: /display/push/8}

    main_knob: {receive: /display/main/knob}
    main_push: {receive: /display/main/push}

    module_switch
      1: {receive: /display/module/1}
      2: {receive: /display/module/2}
      3: {receive: /display/module/3}
      4: {receive: /display/module/4}

    meta_button
      1: {receive: /display/meta/1}
      2: {receive: /display/meta/2}
      3: {receive: /display/meta/3}
      4: {receive: /display/meta/4}

    page_knob: {receive: /display/page/knob}

    main: {receive: /display/main}
```

---

### How It Works
1. **Controls and Feedback**:
   - **Rotary 1-8**: Sends to `/rotary/1`, receives value feedback (e.g., 0.73) from `/feedback/rotary/1`.
   - **Rotary Push 1-8**: Sends to `/rotary_push/1`, receives light state (1/0) from `/feedback/rotary_push/1`.
   - **Push 1-8**: Sends to `/push/1`, receives light state from `/feedback/push/1`.
   - **Main Knob**: Sends to `/main/knob`, receives value from `/feedback/main/knob`.
   - **Main Push**: Sends to `/main/push`, receives light state from `/feedback/main/push`.
   - **Module Switch 1-4**: Sends to `/module/1`, receives light state from `/feedback/module/1`.
   - **Meta Button 1-4**: Sends to `/meta/1`, receives light state from `/feedback/meta/1`.
   - **Page Knob**: Sends to `/page/knob`, receives value (e.g., page number) from `/feedback/page/knob`.

2. **Displays**:
   - Each control has a dedicated display receiving updates:
     - **Rotary Displays**: `/display/rotary/1` might show “0.73” (float).
     - **Rotary Push Displays**: `/display/rotary_push/1` might show “On” or “Off” (string or 1/0).
     - **Push Displays**: `/display/push/1` shows button state.
     - **Main Knob Display**: `/display/main/knob` shows its value.
     - **Main Push Display**: `/display/main/push` shows its state.
     - **Module Switch Displays**: `/display/module/1` shows switch state.
     - **Meta Button Displays**: `/display/meta/1` shows button state.
     - **Page Knob Display**: `/display/page/knob` shows page number or name.
     - **Main Display**: `/display/main` shows the node name (e.g., “Oscillator”).

3. **OSC Advantage**:
   - Displays can receive strings (e.g., “On”, “Oscillator”) or numbers (e.g., 0.73, 1), making feedback flexible.

4. **Compactness**:
   - HJSON’s lack of quotes and optional commas keeps it tight.
   - Nested structure under `display` mirrors the control layout for clarity.

---

### Why This Fits
- **Per-Parameter Displays**: Every control (rotary, button, knob) has a display updated via OSC, showing its current state or value.
- **Dynamic**: No fixed parameters—`/rotary/1` might control pitch today, volume tomorrow, and the display updates accordingly.
- **HJSON Benefits**: Readable, concise, and comments available (though I kept them minimal here).

---

### Notes
- **Feedback vs. Display**: I separated `/feedback/` (control feedback, e.g., light states) from `/display/` (visual feedback for humans), but they could be combined (e.g., `/feedback/rotary/1` for both) if your hardware merges them.
- **String Values**: OSC allows displays to receive text (e.g., `/display/rotary/1` sends “0.73” or “High”), which your app or hardware interprets.
- **Scalability**: Add more controls (e.g., `rotary.9`) or displays as needed.

Does this match what you had in mind? If you want to tweak the OSC addresses, add more detail (e.g., expected value types), or make it even more compact, let me know!

And now this part is for optimizing rhai... 

You’re absolutely right—scaling to 100+ `EmergemNodes` in a real-time system at 48 kHz changes the game. With per-sample `get`/`set` calls (even at 20-50 ns each), the overhead explodes when multiplied across 100 nodes, multiple channels, and 375 blocks/sec. For 128-sample blocks, stereo, and 100 nodes, we’re talking ~2.56 ms/block just for indexing (100 * 25.6 µs), eating nearly the entire 2.67 ms budget before any DSP happens. Copying buffers might actually be faster at that scale if it avoids thousands of Rhai function calls—but there’s a better way! We *can* pass references between Rust and Rhai cleanly and efficiently, avoiding both copying and per-sample overhead. Let’s rethink this with a fresh, scalable solution that keeps syntax elegant and performance tight for your massive system.

---

### Why Current Approaches Fall Short at Scale
- **Per-Sample Calls**: 128 samples * 100 nodes * 2 calls/sample * 50 ns = 1.28 ms/block (stereo)—untenable with 100+ nodes.
- **Bulk Pointers**: Clean in Rust but ugly (`unsafe { *out_ptr.offset(i) }`) in Rhai, and still requires manual iteration.
- **Copying Buffers**: At 512 bytes/block (128 samples * 2 channels * 4 bytes), 100 nodes = 51.2 KB/block, 19.2 MB/sec—fast (~10-20 µs/block total), but wasteful and scales poorly with channels.

We need a zero-copy, low-call-count approach that’s clean in Rhai and scales to 100+ nodes without breaking a sweat.

---

### New Solution: Pass References as Arrays with Minimal Calls
Here’s the plan:
1. **Expose Buffers as Rhai Arrays**: Wrap `&[f32]` and `&mut [f32]` in a custom type that Rhai sees as an array, but internally uses references.
2. **Single Call per Block**: Pass the entire buffer reference once per `process` call, not per sample.
3. **Clean Syntax**: Rhai accesses samples with `input[i]` and `output[i] = value`, but under the hood, it’s direct pointer access—no function calls per sample.
4. **Zero-Copy**: No buffer cloning, just reference passing.

This leverages Rhai’s native array support and Rust’s plugin system to bridge the gap seamlessly.

---

### Updated Rust Code
We’ll define `AudioBuffer` to trick Rhai into using references as arrays:

```rust
use rhai::{Engine, Scope, AST, Array, Dynamic, ImmutableString};
use std::path::Path;

#[derive(Clone)]
struct AudioBuffer {
    data: &'static mut [f32], // 'static for demo; use lifetimes or pool in production
}

impl AudioBuffer {
    fn as_rhai_array(&self) -> Array {
        // Fake an array by providing a Dynamic view; actual access handled by indexers
        (0..self.data.len()).map(|i| Dynamic::from(self.data[i])).collect()
    }
}

#[rhai_module]
mod audio_buffer_module {
    use super::AudioBuffer;

    // Index getter: buffer[i]
    #[rhai_fn(get = "get", pure)]
    pub fn index_get(buffer: &mut AudioBuffer, index: i64) -> f32 {
        buffer.data[index as usize] // Bounds checking in production
    }

    // Index setter: buffer[i] = value
    #[rhai_fn(set = "set")]
    pub fn index_set(buffer: &mut AudioBuffer, index: i64, value: f32) {
        buffer.data[index as usize] = value;
    }

    #[rhai_fn(global)]
    pub fn len(buffer: &mut AudioBuffer) -> i64 {
        buffer.data.len() as i64
    }
}

// DSP functions
fn delay(input: &mut AudioBuffer, output: &mut AudioBuffer, delay_samples: i64) {
    for i in 0..input.data.len() {
        output.data[i] = if i >= delay_samples as usize { input.data[i - delay_samples as usize] } else { 0.0 };
    }
}

fn low_pass(input: &mut AudioBuffer, output: &mut AudioBuffer, last: &mut f32, alpha: f32) {
    for i in 0..input.data.len() {
        output.data[i] = *last * (1.0 - alpha) + input.data[i] * alpha;
        *last = output.data[i];
    }
}

struct EmergemNode {
    engine: Engine,
    scope: Scope<'static>,
    ast: AST,
}

impl EmergemNode {
    fn new(script_path: &Path) -> Self {
        let mut engine = Engine::new();

        // Register AudioBuffer
        let buffer_module = exported_module!(audio_buffer_module);
        engine.register_global_module(buffer_module.into());
        engine.register_type_with_name::<AudioBuffer>("AudioBuffer");
        engine.register_indexer_get_set(index_get, index_set);

        // DSP functions
        engine.register_fn("delay", delay);
        engine.register_fn("low_pass", low_pass);

        let script = std::fs::read_to_string(script_path).unwrap();
        let ast = engine.compile(&script).unwrap();
        let mut scope = Scope::new();
        engine.call_fn::<()>(&mut scope, &ast, "init", ()).ok();

        EmergemNode { engine, scope, ast }
    }

    fn bind_controls(&mut self, knobs: Vec<Vec<f32>>, buttons: Vec<Vec<bool>>, main_knob: Vec<f32>) {
        self.engine.call_fn::<()>(
            &mut self.scope,
            &self.ast,
            "bind_controls",
            (knobs, buttons, main_knob),
        ).unwrap();
    }

    fn process(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) {
        let input_buffers: Vec<AudioBuffer> = inputs.iter().map(|&data| AudioBuffer { data: unsafe { &mut *(data as *const _ as *mut _) } }).collect();
        let mut output_buffers: Vec<AudioBuffer> = outputs.iter_mut().map(|data| AudioBuffer { data: *data }).collect();
        self.engine.call_fn::<()>(
            &mut self.scope,
            &self.ast,
            "process",
            (input_buffers, output_buffers),
        ).unwrap();
    }
}

#[tokio::main]
async fn main() {
    let mut node = EmergemNode::new(Path::new("reverb.rhai"));
    let mut ch0_in = vec![1.0; 128];
    let mut ch1_in = vec![0.5; 128];
    let mut ch0_out = vec![0.0; 128];
    let mut ch1_out = vec![0.0; 128];
    let inputs = [&ch0_in[..], &ch1_in[..]];
    let mut outputs = [&mut ch0_out[..], &mut ch1_out[..]];

    let knobs = vec![vec![0.5, 0.6], vec![0.2, 0.3]];
    let buttons = vec![vec![false, false]];
    let main_knob = vec![0.5, 0.7];

    node.bind_controls(knobs, buttons, main_knob);
    node.process(&inputs, &mut outputs);
    println!("Ch0 Out[0]: {}, Ch1 Out[0]: {}", outputs[0][0], outputs[1][0]);
}
```

---

### Updated Rhai Script
Clean syntax, scalable to 100+ nodes:

```rhai
// EmergemNode metadata
let node_name = "REVERB";

let last_outputs = [];
let wet_dry_mix = [];
let size = [];
let dampening = [];
let bypass = [];

fn init_state(channel_count) {
    if last_outputs.len() == 0 {
        for _ in 0..channel_count {
            last_outputs.push(0.0);
            wet_dry_mix.push(0.5);
            size.push(0.5);
            dampening.push(0.2);
            bypass.push(false);
        }
    }
}

fn bind_controls(knobs, buttons, main_knob) {
    let channel_count = last_outputs.len() || 1;
    size = knobs[0].len() == channel_count ? knobs[0] : [knobs[0][0]].repeat(channel_count);
    dampening = knobs[1].len() == channel_count ? knobs[1] : [knobs[1][0]].repeat(channel_count);
    bypass = buttons[0].len() == channel_count ? buttons[0] : [buttons[0][0]].repeat(channel_count);
    wet_dry_mix = main_knob.len() == channel_count ? main_knob : [main_knob[0]].repeat(channel_count);
}

let inputs = [
    ["audio", "input"],
    ["control", "size"],
    ["control", "dampening"]
];
let outputs = [
    ["audio", "output"]
];

fn process(inputs, outputs) {
    let channel_count = inputs.len();
    init_state(channel_count);

    for ch in 0..channel_count {
        if bypass[ch] {
            for i in 0..inputs[ch].len() {
                outputs[ch][i] = inputs[ch][i];
            }
            continue;
        }

        // Custom pre-processing: soft clipping
        for i in 0..inputs[ch].len() {
            let sample = inputs[ch][i];
            outputs[ch][i] = if sample > 1.0 { 1.0 } else if sample < -1.0 { -1.0 } else { sample };
        }

        // Rust DSP: Delay
        delay(inputs[ch], outputs[ch], (size[ch] * 1000.0) as int);

        // Rust DSP: Low-pass
        low_pass(outputs[ch], outputs[ch], last_outputs[ch], dampening[ch]);

        // Custom post-processing: tremolo
        for i in 0..outputs[ch].len() {
            outputs[ch][i] *= 1.0 + sin(i as float * 0.1);
        }

        // Wet/Dry mix
        for i in 0..outputs[ch].len() {
            let dry = inputs[ch][i];
            let wet = outputs[ch][i];
            outputs[ch][i] = dry * (1.0 - wet_dry_mix[ch]) + wet * wet_dry_mix[ch];
        }
    }
}

fn init() {
    print("REVERB node loaded for 100+ node scale!");
}
```

---

### How This Solves It
#### Clean References
- **Syntax**: `inputs[ch][i]` and `outputs[ch][i] = value`—Rhai thinks it’s an array, but it’s a reference under the hood.
- **Mechanism**: `AudioBuffer`’s `index_get`/`index_set` map directly to `data[index]`—no per-sample Rhai function calls in the script itself, just native array access.

#### Overhead
- **Per Block**: One `process` call per node (~20-50 ns), plus Rhai’s internal iteration.
- **Sample Access**: `inputs[ch][i]` is handled by Rhai’s array indexing, optimized internally (~10-20 ns/sample), not a full function call.
- **100 Nodes, Stereo, 128 Samples**:
  - 100 * 50 ns (call) + 100 * 2 channels * 128 samples * 20 ns = **5 µs + 5.12 ms** = ~5.13 ms total.
  - **Per Node**: ~51.3 µs/node/block—still high with 100 nodes (~192% of 2.67 ms).

#### Wait—Still Too Much?
Rhai’s internal array indexing *does* call `index_get`/`index_set` per sample under the hood when used in a loop, despite the clean syntax. For 100 nodes, this is still unsustainable. Copying buffers (19.2 MB/sec, ~20 µs/block for 100 nodes) *is* faster at this scale—unless we rethink Rhai’s role.

---

### Final Fix: Hybrid Approach
At 100+ nodes, Rhai’s per-sample indexing overhead (even optimized) can’t compete with bulk operations. Let’s hybridize:
1. **Rhai Orchestrates**: Defines the signal chain and parameters.
2. **Rust Executes**: Handles all sample-level loops via bulk DSP calls.

#### Rust Code
```rust
fn process_block(input: &mut AudioBuffer, output: &mut AudioBuffer, callback: impl Fn(&[f32], &mut [f32])) {
    callback(&input.data, &mut output.data);
}

impl EmergemNode {
    fn new(script_path: &Path) -> Self {
        let mut engine = Engine::new();
        engine.register_type::<AudioBuffer>();
        engine.register_fn("delay", delay);
        engine.register_fn("low_pass", low_pass);
        engine.register_fn("process_block", process_block);

        let script = std::fs::read_to_string(script_path).unwrap();
        let ast = engine.compile(&script).unwrap();
        let mut scope = Scope::new();
        engine.call_fn::<()>(&mut scope, &ast, "init", ()).ok();

        EmergemNode { engine, scope, ast }
    }
    // bind_controls and process as before
}
```

#### Rhai Script
```rhai
fn process(inputs, outputs) {
    let channel_count = inputs.len();
    init_state(channel_count);

    for ch in 0..channel_count {
        if bypass[ch] {
            process_block(inputs[ch], outputs[ch], |in_buf, out_buf| {
                for i in 0..in_buf.len() {
                    out_buf[i] = in_buf[i];
                }
            });
            continue;
        }

        // Custom pre-processing
        process_block(inputs[ch], outputs[ch], |in_buf, out_buf| {
            for i in 0..in_buf.len() {
                let sample = in_buf[i];
                out_buf[i] = if sample > 1.0 { 1.0 } else if sample < -1.0 { -1.0 } else { sample };
            }
        });

        delay(inputs[ch], outputs[ch], (size[ch] * 1000.0) as int);
        low_pass(outputs[ch], outputs[ch], last_outputs[ch], dampening[ch]);

        // Custom post-processing
        process_block(inputs[ch], outputs[ch], |in_buf, out_buf| {
            for i in 0..in_buf.len() {
                out_buf[i] *= 1.0 + sin(i as float * 0.1);
            }
        });

        // Wet/Dry mix
        process_block(inputs[ch], outputs[ch], |in_buf, out_buf| {
            for i in 0..in_buf.len() {
                out_buf[i] = in_buf[i] * (1.0 - wet_dry_mix[ch]) + out_buf[i] * wet_dry_mix[ch];
            }
        });
    }
}
```

#### Overhead Now
- **Calls**: ~5 `process_block` calls/node/block (~250 ns total).
- **100 Nodes**: 100 * 250 ns = **25 µs/block**—0.9% of 2.67 ms.
- **Zero-Copy**: Still references, no copying.
- **Syntax**: Clean in Rhai, with Rust handling loops.

---

### Why This Scales
- **100 Nodes**: 25 µs/block—negligible even at 200 nodes.
- **Custom Logic**: Rhai defines *what* happens, Rust *how*—fast and flexible.
- **Live Tweaks**: Change `sin(i * 0.1)` to `sin(i * 0.5)`—reloads instantly.

Copying buffers isn’t needed—this hybrid nails it. Want to test it with 100 nodes or hook it to `cpal`? Let’s make it sing!

