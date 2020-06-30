(ns clj-ml.nneighbors
)
i

(defn prebuilt-distance-fn
  [fn-name]
  (case (keyword fn-name)
    :xy_map
      (fn [{x0 :x y0 :y} {x1 :x y1 :y}]
        (Math/sqrt
          (+
            (Math/pow (- x1 x0) 2.0)
            (Math/pow (- y1 y0) 2.0))))))

(defn build
  [distance-fn
   & {:keys [weight-fn]}]
  (let [distance-fn (if (fn? distance-fn)
                      distance-fn
                      (prebuilt-distance-fn distance-fn))]
  ))
