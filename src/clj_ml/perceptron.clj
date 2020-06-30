(ns clj-ml.perceptron
  (:require [clojure.core.matrix :as matrix]
            [clojure.string :as clj-str])
  (:import [java.lang.Math]))



(defn activation-fn
  [perceptron]
  (case (keyword (:activation perceptron))
    :tanh
      #(Math/tanh %)
    :exp
      #(/ 1 (inc (Math/exp (* -1 %))))
    nil))

(defn activation-fn-slope
  [perceptron]
  (case (keyword (:activation perceptron))
    :tanh
      #(let [n (Math/tanh %)]
         (- 1.0 (* n n)))
    :exp
      #(let [n (Math/exp %)]
         (/ n (Math/pow (inc n) 2)))
    nil))

(defn output-range
  [perceptron]
  (case (keyword (:activation perceptron))
    :tanh [-1 1]
    :exp [0 1]
    nil))


(defn- random-weights
  [input-nodes output-nodes]
  (let [weights (matrix/new-matrix :vectorz input-nodes output-nodes)]
    (dorun
      (map
        (fn [input-node]
          (dorun
          (map
            (fn [output-node]
              (matrix/mset!
                weights
                input-node
                output-node
                (dec (* 2 (rand)))))
            (range output-nodes))))
        (range input-nodes)))
    weights))

(defn build
  [layers
   & {:keys [train-rate activation output-nodes]
      :or {train-rate 0.0001
           activation "tanh"
           output-nodes 1}}]
  (let [node-counts (conj layers output-nodes)]
    {:weights
       (loop [[layer-nodes & other-nodes] node-counts
              weights []]
         (if (empty? other-nodes)
           weights
           (recur
             other-nodes
             (conj
               weights
               (random-weights
                 (inc layer-nodes) ;; include bias
                 (first other-nodes))))))
     ;; total weighted signals transmitted to nodes
     :inputs
       (map
         (fn [layer-nodes]
           (matrix/new-vector :vectorz layer-nodes))
         ;; input layer is set directly and does not need an input vector
         (rest node-counts))
     ;; signals emitted from nodes, including bias nodes
     ;; linear for the input layer, sigmoid for hidden layers and output node
     :outputs
       (map
         (fn [layer-nodes]
           (matrix/new-vector :vectorz (inc layer-nodes)))
         node-counts)
     :train-rate train-rate
     :activation (name activation)}))


(defn forward-pass
  [{:keys [weights inputs outputs] :as perceptron} input-signals]
  (let [act-fn (activation-fn perceptron)]
    ;; set input layer signals
    (matrix/assign! (first outputs) (cons 1 input-signals))
    ;; pass forward through layers
    (loop [[output & other-outputs] outputs
           [input & other-inputs] inputs
           [weight & other-weights] weights]
      (when-not (empty? other-outputs)
        ;; output * weights -> input
        (matrix/assign! input (matrix/mmul output weight))
        ;; act-fn( input ) -> output
        (matrix/assign! (first other-outputs) (cons 1 (matrix/emap act-fn input)))
        (recur other-outputs other-inputs other-weights)))
    (rest (last outputs))))


(defn backward-prop
  [{:keys [inputs outputs weights train-rate] :as perceptron} output-errors]
  (let [grad-fn (activation-fn-slope perceptron)]
    ;; set output-layer error gradient
    (matrix/assign!
      (last inputs)
      (matrix/emap
        (fn [output-input output-error]
          (* output-error (grad-fn output-input)))
        (last inputs)
        output-errors))
    ;; pass backward through layers
    (loop [[input & other-inputs] (reverse inputs)
           [output & other-outputs] (rest (reverse outputs))
           [weight & other-weights] (reverse weights)]
      (when weight
        ;; propagate errors
        (when other-inputs
          (matrix/assign!
            (first other-inputs)
            (matrix/emap
              (fn [node-input backprop]
                (*
                  (grad-fn node-input)
                  backprop))
              (first other-inputs)
              (rest (matrix/mmul weight input)))))
        ;; update weights
        (matrix/add!
          weight
          (matrix/scale
            (matrix/outer-product output input)
            train-rate))
        (recur other-inputs other-outputs other-weights)))))


(defn train
  [perceptron data & {:keys [log-fn cycles]}]
  (loop [ind 0]
    (when (or (not cycles) (< ind cycles))
      (when log-fn (log-fn perceptron ind))
      (doseq [{:keys [input value]} (shuffle data)]
        (let [[prediction] (forward-pass perceptron input)
              error (- value prediction)]
          (backward-prop perceptron [error])))
      (recur (inc ind)))))



;; =============================================================================
;;  Demo
;; =============================================================================

(defn- print-circle-test
  [perceptron iter]
  (let [[neg-val pos-val] (output-range perceptron)]
    (println
      (str
       "\n\n"
       "Training Cycles: "
       iter
       "\n"
       (clj-str/join
         "\n"
         (map
           (fn [y]
             (clj-str/join
               " "
               (map
                 (fn [x]
                   (if (>
                         (first (forward-pass perceptron [x y]))
                         (+ neg-val (/ (- pos-val neg-val) 2.0)))
                     "0"
                     "*"))
                 (range -1 1 0.05))))
           (range -1 1 0.05)))
       "\n\n"))))

(defn- make-circle-data
  [n-pnts noise neg-val pos-val circle-size]
  (repeatedly
    n-pnts
    (fn []
      (let [x (dec (* 2 (rand)))
            y (dec (* 2 (rand)))]
        {:input [x y]
         :value (if (< (rand) noise)
                  (if (< (rand) 0.5) neg-val pos-val)
                  (let [radius (+ (* x x) (* y y))]
                    (if (> radius circle-size) neg-val pos-val)))}))))

(defn circle-test
  [& {:keys [act-fn train-rate noise n-pnts circle-size]
      :or {noise 0.25
           n-pnts 25000
           circle-size 0.8}}]
  (let [act-fn (or act-fn :tanh)
        train-rate (or
                     train-rate
                     (case (keyword act-fn)
                       :tanh 0.0005
                       :exp 0.005
                       nil))
        perceptron (build
                     [2 4]
                     :train-rate train-rate
                     :activation act-fn)
        [neg-val pos-val] (output-range perceptron)
        data (make-circle-data n-pnts noise neg-val pos-val circle-size)]
    (train
      perceptron
      data
      :log-fn print-circle-test)))
