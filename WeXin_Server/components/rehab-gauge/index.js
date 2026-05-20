Component({
  properties: {
    score: {
      type: Number,
      value: 0,
      observer: '_draw'
    },
    size: {
      type: Number,
      value: 150
    }
  },

  data: {
    canvasId: 'gauge_' + Math.random().toString(36).slice(2, 8)
  },

  lifetimes: {
    attached() {
      setTimeout(() => this._draw(), 200);
    }
  },

  methods: {
    _draw() {
      const id = '#' + this.data.canvasId;
      const query = this.createSelectorQuery();
      query.select(id).fields({ node: true, size: true }).exec((res) => {
        if (!res[0] || !res[0].node) return;
        const canvas = res[0].node;
        const ctx = canvas.getContext('2d');
        const dpr = wx.getSystemInfoSync().pixelRatio;
        const s = this.properties.size;
        canvas.width = s * dpr;
        canvas.height = (s * 0.55 + 30) * dpr;
        ctx.scale(dpr, dpr);

        const score = Math.min(this.properties.score, 100);
        const color = score < 50 ? '#e74c3c' : score < 70 ? '#f39c12' : '#27ae60';
        const cx = s / 2, cy = s / 2, r = s / 2 - 10;

        // 背景弧
        ctx.beginPath();
        ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
        ctx.strokeStyle = '#e8e8e8';
        ctx.lineWidth = 10;
        ctx.lineCap = 'round';
        ctx.stroke();

        // 前景弧
        ctx.beginPath();
        ctx.arc(cx, cy, r, Math.PI, Math.PI + (score / 100) * Math.PI);
        ctx.strokeStyle = color;
        ctx.stroke();

        // 分数
        ctx.fillStyle = color;
        ctx.font = 'bold ' + Math.round(s * 0.22) + 'px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(Math.round(score).toString(), cx, cy - 6);
      });
    }
  }
});
