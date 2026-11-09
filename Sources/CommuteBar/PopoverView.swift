import AppKit
import Charts
import SwiftUI

struct CommutePopover: View {
    init(model: CommuteModel, onSettings: @escaping () -> Void = {}, previewHover: Double? = nil, previewMode: String? = nil) {
        self.model = model
        self.onSettings = onSettings
        _hoveredMinute = State(initialValue: previewHover)
        if let previewMode { model.chartMode = previewMode }
    }
    @ObservedObject var model: CommuteModel
    var onSettings: () -> Void = {}
    // chartMode lives on the model (see CommuteModel.chartMode); @State is written out by hand here because
    // the Command Line Tools lack the SwiftUIMacros plugin that expands @State.
    private var chartMode: String {
        get { model.chartMode }
        nonmutating set { model.chartMode = newValue }
    }
    private var _hoveredMinute = State<Double?>(initialValue: nil)
    private var hoveredMinute: Double? {
        get { _hoveredMinute.wrappedValue }
        nonmutating set { _hoveredMinute.wrappedValue = newValue }
    }
    private var _hoveredDay = State<Date?>(initialValue: nil)
    private var hoveredDay: Date? {
        get { _hoveredDay.wrappedValue }
        nonmutating set { _hoveredDay.wrappedValue = newValue }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Label("Commute", systemImage: "car.fill").font(.headline)
                Spacer()
                Button {
                    model.refresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh traffic").disabled(model.refreshing)
                Button(action: onSettings) { Image(systemName: "gearshape") }.help("Settings")
            }.buttonStyle(.plain)
            Picker("Direction", selection: $model.selection) {
                Text("Auto").tag("auto")
                Text("To work").tag("to_work")
                Text("To home").tag("to_home")
            }.pickerStyle(.segmented).labelsHidden()
            if let route = model.route {
                HStack(alignment: .firstTextBaseline, spacing: 5) {
                    if let latest = route.latest {
                        Text("\(Int(latest.minutes))").font(.system(size: 32, weight: .semibold, design: .rounded))
                        Text("min to \(model.destination.lowercased())").foregroundStyle(.secondary)
                        Spacer()
                        Text("\(clockTime(latest.date))\(Date().timeIntervalSince(latest.date) > 900 ? " · stale" : "")")
                            .font(.caption).foregroundStyle(Date().timeIntervalSince(latest.date) > 900 ? Color.orange : Color.secondary)
                            .help("Sampled \(latest.date.formatted())")
                    } else {
                        Text("Waiting for traffic…").foregroundStyle(.secondary)
                    }
                }
                VStack(alignment: .leading, spacing: 3) {
                    recommendation(route)
                    if let offset = route.forecastOffsetMinutes, abs(offset) >= 1 {
                        // An offset beyond ~1.5× the model's day-to-day swing means today is an outlier —
                        // the suggestion is still the best guess, but flag that it rests on thin ground.
                        let tau = route.model?.tau ?? 0
                        let unusual = tau > 0 && abs(offset) >= 1.5 * tau
                        Text(
                            "Today is running \(offset > 0 ? "+" : "−")\(Int(abs(offset).rounded())) min vs the usual\(unusual ? " · unusual day" : "")"
                        )
                        .font(.caption).foregroundStyle(unusual ? Color.orange : Color.secondary)
                        .help(
                            (route.showsModel
                                ? "Today's recent samples compared with the usual drive for the same time of day. The prediction already accounts for this and fades it out over the next few hours."
                                : "Today's recent samples compared with the usual drive for the same time of day. The dashed line and the expected time adjust for it and fade it out over the next few hours.")
                                + (unusual ? " This deviation is unusually large — treat the suggestion as less certain." : ""))
                    }
                    if let rain = route.rain {
                        Text(
                            rain.src == "obs"
                                ? "Rain \(decimal(rain.mm))mm today — already in the forecast"
                                : "Rain ~\(decimal(rain.mm))mm expected \(dayPhrase(rain.date)) — already in the forecast"
                        )
                        .font(.caption).foregroundStyle(.secondary)
                        .help(
                            "Precipitation on the forecast day from the weather feed. The model learned from past wet days how much each mm adds to the drive."
                        )
                    }
                }
                Divider()
                Picker("Chart", selection: $model.chartMode) {
                    Text(route.model?.label?.split(separator: " ").first.map(String.init) ?? "Predicted").tag("typical")
                    Text("Today").tag("today")
                    Text("Model").tag("model")
                }.pickerStyle(.segmented).labelsHidden()
                if chartMode == "typical" { typicalChart(route) } else if chartMode == "today" { todayChart() } else { modelTab(route) }
                HStack {
                    Text(
                        chartMode == "typical"
                            ? "\(route.modelLabel)\(route.curveDate.map { " · \(dayLabel($0))" } ?? "") · shaded \(route.intervalPercent)% range"
                            : chartMode == "today"
                                ? "Today · both directions"
                                : "\(route.modelLabel) · accuracy record"
                    ).foregroundStyle(.secondary)
                        .help(
                            chartMode == "typical"
                                ? (route.showsModel
                                    ? "Predicted drive time for each departure time, learned from your own commute history and today's traffic so far. The shaded band is an \(route.intervalPercent)% range: about 8 in 10 actual drive times should land inside it. It forecasts what Google Maps will show, and it does not anticipate incidents. The dashed curve and faint band are the forecast as of this morning, before today's traffic was seen — what the day was expected to look like."
                                    : "The simple typical drive time for each departure time: a smoothed average of recent samples from the same kind of day, shifted to match how today is running once the day is under way. The shaded band is the middle \(route.intervalPercent)% of those past drives, a plain summary rather than a checked forecast. The Predicted model takes over once it has been scored on enough days.")
                                : chartMode == "model"
                                    ? "How the model has actually done, day by day: how far its expected drive times missed the real ones, and how often the real drive landed inside its stated range. The dashed line is the simple curve it must keep beating to stay on screen."
                                    : "")
                    Spacer()
                    Button("Maps ↗") { if let url = URL(string: route.url) { NSWorkspace.shared.open(url) } }.buttonStyle(.plain)
                }.font(.caption)
            } else if model.needsSetup {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Welcome to Commute Bar").font(.headline)
                    Text("Enter your home and work addresses to start tracking your drive.").font(.caption).foregroundStyle(.secondary)
                    Button("Open Settings", action: onSettings).buttonStyle(.borderedProminent).controlSize(.small)
                }.padding(.vertical, 12)
            } else {
                Text(model.locationText).font(.callout).foregroundStyle(.secondary)
                Text("Select a direction to see your commute.").font(.caption).foregroundStyle(.secondary).padding(.vertical, 16)
            }
            if let error = model.error {
                Text(error).font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            }
            Divider()
            HStack {
                Text(model.selection == "auto" ? model.locationText : "Direction selected manually")
                    .font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    .help(model.samplerText)
                Spacer()
                Button("Quit") { NSApplication.shared.terminate(nil) }.font(.caption).buttonStyle(.plain)
            }
        }
        .padding(16).frame(width: 370).background(Color(nsColor: .windowBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
        .onChange(of: chartMode) { _ in
            hoveredMinute = nil
            hoveredDay = nil
        }
        .onChange(of: model.direction) { _ in
            hoveredMinute = nil
            hoveredDay = nil
        }
    }
    private func durationText(_ minutes: Int) -> String {
        minutes >= 60 ? "\(minutes / 60)h\(minutes % 60 == 0 ? "" : " \(minutes % 60)m")" : "\(minutes)m"
    }
    @ViewBuilder private func recommendation(_ route: Route) -> some View {
        let rec = route.shownRecommendation
        if let dateText = rec.departAt, let date = parseDate(dateText), let typical = rec.minutes {
            let minutes = rec.expectedMinutes ?? typical
            // Pessimistic arrival: leave at the suggested time, get there by this even on the slow side of the band.
            let departMinute = Int(localMinute(date))
            let slowArrive = route.history.min { abs($0.minuteOfDay - departMinute) < abs($1.minuteOfDay - departMinute) }
                .map { "arrive ~\(clockTime(date.addingTimeInterval($0.p90Minutes * 60))) if it's slow" }
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(
                        rec.forDate.map { "\(dayLabel($0)): \(rec.isLatest ? "leave by around" : "leave around") \(clockTime(date))" }
                            ?? "\(rec.isLatest ? "Leave by around" : "Leave around") \(clockTime(date))"
                    ).font(.subheadline.weight(.medium))
                    Spacer()
                    Text("~\(minutes.formatted(.number.precision(.fractionLength(0...1)))) min").font(.subheadline).foregroundStyle(
                        .secondary)
                }
                if let fastest = rec.fastestMinutes, let fastestText = rec.fastestDepartAt, let fastestDate = parseDate(fastestText) {
                    let later = Int(date.timeIntervalSince(fastestDate) / 60)  // negative: earlier than the fastest time
                    let shift = rec.isLatest || rec.isMiddle ? abs(later) : max(0, -later)
                    let cost =
                        typical - fastest < 0.05
                        ? "No extra driving"
                        : "\((typical - fastest).formatted(.number.precision(.fractionLength(0...1)))) min more driving"
                    let line =
                        (shift > 0
                            ? "\(cost) · \(durationText(shift)) \(later > 0 ? "later" : "earlier") than the fastest time"
                            : rec.isMiddle
                                ? "The fastest time, and near the middle of your window"
                                : "\(rec.isLatest ? "Latest" : "Earliest") time within your driving allowance")
                        + (slowArrive.map { " · \($0)" } ?? "")
                    Text(line).font(.caption).foregroundStyle(.secondary)
                } else if let slowArrive {
                    Text(slowArrive).font(.caption).foregroundStyle(.secondary)
                }
            }.help(
                rec.isMiddle
                    ? "The time closest to the middle of your window that avoids the rush without much extra driving (\(rec.sampleCount) days of history behind it). A bigger driving allowance in Settings favors the middle more; zero picks the fastest times. \(rec.forDate != nil ? "This is the forecast made before that day's traffic is known; it sharpens once the day gets going." : "Adjusted for how today is running.")"
                    : rec.forDate != nil
                        ? "\(rec.strategyPhrase) \(rec.extraDriveMinutes.formatted()) minutes of the fastest drive in that window, based on \(rec.sampleCount) days. This is the forecast made before that day's traffic is known; it sharpens once the day gets going. Adjust your allowance in Settings."
                        : "\(rec.strategyPhrase) \(rec.extraDriveMinutes.formatted()) minutes of the fastest remaining drive, based on \(rec.sampleCount) days. Adjust your allowance in Settings. Historical estimate, adjusted for how today is running; not a traffic forecast."
            )
        } else {
            Text(statusText(rec.status)).font(.caption).foregroundStyle(.secondary)
        }
    }
    private func statusText(_ status: String) -> String {
        switch status {
        case "no_data": return "More history needed for a departure estimate"
        case "weekend": return "No commute on weekends"
        case "day_off":
            let reason = model.snapshot?.dayOffReason.flatMap { $0 == "Day off" ? nil : $0 }
            return "Day off\(reason.map { " · \($0)" } ?? "") · no departure suggestion"
        default: return "No departure window remaining today"
        }
    }
    private struct PlotPoint: Identifiable {
        let id: Int
        let minute: Double
        let minutes: Double
        let days: Int?
        let low: Double
        let high: Double
        let rangeText: String  // whole minutes when the model supplies them
        let minimum: Double?
        let maximum: Double?
    }
    /// One vertex of the forward-looking curve and its shaded range (which fans out from the red dot while the window is open).
    private struct FanPoint: Identifiable {
        let id: Int
        let minute: Double
        let minutes: Double
        let low: Double
        let high: Double
    }
    /// How far past the newest sample the range takes to reach its full width.
    private static let fanMinutes = 60.0
    private struct TodayPoint: Identifiable {
        let id: String
        let label: String
        let minute: Double  // real local minute of day
        let x: Double  // position on the day axis (the real minute of day)
        let minutes: Double
    }
    /// Today's number line: real clock time from the earliest window start to the latest window end, one tick an hour.
    private struct DayAxis {
        let domain: ClosedRange<Double>
        init(_ routes: [Route]) {
            let lower = Double(routes.map(\.windowStart).min() ?? 0), upper = Double(routes.map(\.windowEnd).max() ?? 1440)
            domain = lower...max(upper, lower + 1)
        }
        /// An hourly tick for every whole hour in the domain, labeled every second hour (and at both ends), so labels never collide.
        var ticks: [(minute: Double, labeled: Bool)] {
            let first = ceil(domain.lowerBound / 60) * 60
            let hours = stride(from: first, through: domain.upperBound, by: 60).map { minute in
                (minute: minute, labeled: Int((minute - first) / 60) % 2 == 0)
            }
            return hours
        }
    }
    private static let directionLabels = [("to_work", "To work"), ("to_home", "To home")]
    /// The prediction to draw. The simple curve carries today's adjustment as a separate forecast; that forecast *is* the
    /// prediction, so the curve and its range are shifted by it here and nothing else is drawn for it. (The learned model
    /// already includes today's pace, so it has no separate forecast and is drawn as is.)
    private func points(_ route: Route) -> [PlotPoint] {
        let adjusted = Dictionary(route.forecastPoints.map { ($0.minuteOfDay, $0.minutes) }, uniquingKeysWith: { _, new in new })
        return route.history.map { point in
            let shift = (adjusted[point.minuteOfDay] ?? point.predicted) - point.predicted
            let low = point.p10Minutes + shift, high = point.p90Minutes + shift
            let rangeText =
                (point.rangeLow != nil && point.rangeHigh != nil)
                ? "\(point.rangeLow!)–\(point.rangeHigh!)" : "\(decimal(low))–\(decimal(high))"
            return PlotPoint(
                id: point.minuteOfDay, minute: Double(point.minuteOfDay), minutes: point.predicted + shift, days: point.dayCount, low: low,
                high: high, rangeText: rangeText, minimum: point.minMinutes, maximum: point.maxMinutes)
        }
    }
    /// This direction's in-window samples today, oldest first, one per minute (a repeated x would break the smoothed line).
    private func actualToday(_ route: Route) -> [TodayPoint] {
        var byMinute: [Double: Observation] = [:]
        for observation in route.today { byMinute[observation.minuteOfDay] = observation }
        return byMinute.keys.sorted().map {
            TodayPoint(id: "actual-\(Int($0))", label: "Today", minute: $0, x: $0, minutes: byMinute[$0]!.minutes)
        }
    }
    /// Runs of samples with no hole longer than `maxGap` minutes (samples come every 10).
    private func actualSegments(_ points: [TodayPoint], maxGap: Double = 25) -> [[TodayPoint]] {
        var segments: [[TodayPoint]] = []
        for point in points {
            if let last = segments.last?.last, point.x - last.x <= maxGap {
                segments[segments.count - 1].append(point)
            } else {
                segments.append([point])
            }
        }
        return segments
    }
    private func todayPoints(_ axis: DayAxis) -> [TodayPoint] {
        Self.directionLabels.flatMap { key, label -> [TodayPoint] in
            // Two samples in the same minute (a refresh landing on a scheduled sample) would make the smoothed line
            // blow up at the repeated x, so keep the newest of each minute.
            var byMinute: [Double: Observation] = [:]
            let route = model.snapshot?.routes[key]
            for observation in route?.todayAll ?? route?.today ?? [] where axis.domain.contains(observation.minuteOfDay) {
                byMinute[observation.minuteOfDay] = observation
            }
            return byMinute.keys.sorted().map { minute in
                let observation = byMinute[minute]!
                return TodayPoint(id: "\(key)-\(Int(minute))", label: label, minute: minute, x: minute, minutes: observation.minutes)
            }
        }
    }
    /// A y range that fits the data with some air, rounded to tidy minute values, instead of always starting at zero.
    /// A flat commute zooms in; a swingy one widens. With nothing to plot it falls back to 0...30.
    private func yDomain(_ values: [Double]) -> ClosedRange<Double> {
        guard let low = values.min(), let high = values.max() else { return 0...30 }
        let pad = max(2, (high - low) * 0.15)
        let step = (high - low) + 2 * pad > 20 ? 5.0 : 2.0
        let floor = max(0, ((low - pad) / step).rounded(.down) * step)
        let ceiling = max(((high + pad) / step).rounded(.up) * step, floor + 8)
        return floor...ceiling
    }
    private func timeLabel(_ minute: Double) -> String {
        let m = Int(minute)
        return String(format: "%d:%02d", m / 60, m % 60)
    }
    private func decimal(_ value: Double) -> String { value.formatted(.number.precision(.fractionLength(0...1))) }
    /// dayLabel for mid-sentence use: "today" and "tomorrow" lowercased, weekday names unchanged.
    private func dayPhrase(_ iso: String) -> String {
        let label = dayLabel(iso)
        return label == "Today" || label == "Tomorrow" ? label.lowercased() : label
    }
    /// A percent with a true minus sign ("−5%"), so a weight that turned negative never reads as positive.
    private func percentText(_ percent: Double) -> String {
        let p = Int(percent.rounded())
        return p < 0 ? "−\(-p)%" : "\(p)%"
    }
    /// Fixed one decimal, for scores read side by side ("1.0 · 1.1"), where a dropped trailing zero looks like a different precision.
    private func oneDecimal(_ value: Double) -> String { value.formatted(.number.precision(.fractionLength(1))) }
    /// Regular ticks, with labels thinned so a long domain stays readable.
    private func xAxis(_ range: ClosedRange<Double>) -> some AxisContent {
        let span = range.upperBound - range.lowerBound
        let step = span <= 120 ? 15 : span <= 480 ? 30 : 60
        // Whole-hour spacing with at most five labels, since the edge labels hang inward and need room.
        let labelStep = [step, 60, 120, 180, 240].first { $0 % step == 0 && span / Double($0) <= 5 } ?? 240
        let firstTick = Int(ceil(range.lowerBound / Double(step))) * step
        let ticks = stride(from: firstTick, through: Int(range.upperBound), by: step).map(Double.init)
        return AxisMarks(values: ticks) { value in
            AxisTick()
            if let minute = value.as(Double.self), (Int(minute) - firstTick) % labelStep == 0 {  // anchored so the first tick is labeled
                AxisGridLine().foregroundStyle(.secondary.opacity(0.15))
                // Edge labels hang inward; centered ones would overflow the plot and be dropped.
                AxisValueLabel(anchor: minute == range.lowerBound ? .topLeading : minute == range.upperBound ? .topTrailing : .top) {
                    Text(timeLabel(minute)).font(.system(size: 9, design: .monospaced)).fixedSize()
                }
            }
        }
    }
    private func hoverCapture(_ proxy: ChartProxy) -> some View {
        GeometryReader { geometry in
            Rectangle().fill(Color.clear).contentShape(Rectangle())
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let location):
                        let frame = geometry[proxy.plotAreaFrame]
                        hoveredMinute = frame.contains(location) ? proxy.value(atX: location.x - frame.minX, as: Double.self) : nil
                    case .ended: hoveredMinute = nil
                    }
                }
        }
    }
    /// The same overlay for the report-card charts, whose x axis is scored days rather than minutes of day.
    private func modelHoverCapture(_ proxy: ChartProxy) -> some View {
        GeometryReader { geometry in
            Rectangle().fill(Color.clear).contentShape(Rectangle())
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let location):
                        let frame = geometry[proxy.plotAreaFrame]
                        hoveredDay = frame.contains(location) ? proxy.value(atX: location.x - frame.minX, as: Date.self) : nil
                    case .ended: hoveredDay = nil
                    }
                }
        }
    }
    private func departureMinute(_ date: Date) -> Double { localMinute(date) }
    @ViewBuilder private func typicalChart(_ route: Route) -> some View {
        let data = points(route)
        let range = Double(route.windowStart)...Double(route.windowEnd)
        // Today's actual drive times (morning and evening alike), kept after the window closes; the newest one gets a dot.
        let windowOpen = range.contains(localMinute(Date()))
        // Today's samples are drawn when the curve is for today, or when today is a normal day (its actuals still compare with
        // tomorrow's similar curve). After a weekend's window the curve is the next normal day's, which a Saturday says nothing
        // about: splicing them together would jump between two different kinds of day.
        let curveIsToday = route.curveDate.map { $0 == localDay(Date()) } ?? true
        let showsToday = curveIsToday || (model.snapshot?.dayStatus ?? "weekday") == "weekday"
        let actual = showsToday ? actualToday(route) : []
        let splitting = windowOpen && !actual.isEmpty
        // While the window is open the blue curve starts at the red dot, for the simple typical curve as well as the learned
        // one: what already happened is superseded by the actual line, so blue only covers what is still ahead. Its range
        // fans out from zero width there, since the further ahead, the less certain.
        let aheadPoints = splitting ? data.filter { $0.minute > actual.last!.x } : data
        let fan: [FanPoint] = {
            guard splitting, let newest = actual.last else {
                return data.map { FanPoint(id: $0.id, minute: $0.minute, minutes: $0.minutes, low: $0.low, high: $0.high) }
            }
            let anchor = FanPoint(id: -1, minute: newest.x, minutes: newest.minutes, low: newest.minutes, high: newest.minutes)
            return [anchor]
                + aheadPoints.map { point in
                    let spread = min(1, ((point.minute - newest.x) / Self.fanMinutes).squareRoot())
                    return FanPoint(
                        id: point.id, minute: point.minute, minutes: point.minutes,
                        low: point.minutes - (point.minutes - point.low) * spread,
                        high: point.minutes + (point.high - point.minutes) * spread)
                }
        }()
        // The nearest actual sample within arm's reach (samples land every ~10 min): dragging over the red
        // line reads what actually happened against what the model expected at that minute.
        let selectedActual: TodayPoint? = hoveredMinute.flatMap { m in
            actual.min { abs($0.x - m) < abs($1.x - m) }.flatMap { abs($0.x - m) <= 6 ? $0 : nil }
        }
        let selected: PlotPoint? =
            selectedActual != nil
            ? nil
            : hoveredMinute.flatMap { m in
                // In the observed past there is no forward curve to read — the actual line owns that region.
                guard !splitting || m > actual.last!.x else { return nil }
                return aheadPoints.min { abs($0.minute - m) < abs($1.minute - m) }
            }
        // The morning-of forecast, made before any of today's traffic was seen: its faint band is where the day
        // was expected to land and the dashed line is what it was expected to do, drawn for the whole window so
        // today's red line can be read against the expectation rather than alone.
        let thought: [FanPoint] = (splitting ? route.base ?? [] : []).map { point in
            FanPoint(
                id: point.id, minute: Double(point.minuteOfDay), minutes: point.typicalMinutes,
                low: point.p10Minutes, high: point.p90Minutes)
        }
        VStack(alignment: .leading, spacing: 4) {
            VStack(alignment: .leading, spacing: 2) {
                if let selected = selectedActual {
                    let expected = data.min { abs($0.minute - selected.minute) < abs($1.minute - selected.minute) }
                    let versus = curveIsToday ? "expected" : "\(route.curveDate.map(dayPhrase) ?? "tomorrow")'s forecast"
                    Text(
                        "\(timeLabel(selected.minute)) · actual \(decimal(selected.minutes)) min\(expected.map { " · \(versus) ~\(decimal($0.minutes)) min" } ?? "")"
                    )
                    if let expected {
                        let delta = selected.minutes - expected.minutes
                        Text(
                            "\(delta >= 0 ? "+" : "−")\(decimal(abs(delta))) min vs \(versus) · \(route.intervalPercent)% range \(expected.rangeText) min"
                        ).font(.caption2)
                    }
                } else if let selected {
                    let all =
                        selected.minimum.flatMap { lo in selected.maximum.map { hi in " · all \(lo.formatted())–\(hi.formatted()) min" } }
                        ?? ""
                    Text(
                        "\(timeLabel(selected.minute)) · ~\(decimal(selected.minutes)) min · \(route.intervalPercent)% range \(selected.rangeText) min"
                    )
                    Text(
                        "arrive ~\(timeLabel(selected.minute + selected.minutes.rounded()))\(selected.days.map { " · \($0) days" } ?? "")\(all)"
                    ).font(.caption2)
                } else {
                    Text("Drive time · hover for details")
                }
            }.font(.caption).foregroundStyle(.secondary).frame(height: 32, alignment: .topLeading)
            if data.isEmpty {
                Text(
                    route.modelState == "collecting"
                        ? "Collecting data — predictions start after a few commuting days" : "No samples in this range yet"
                )
                .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(height: 145).frame(maxWidth: .infinity)
            } else {
                Chart {
                    ForEach(thought) { point in
                        AreaMark(
                            x: .value("Departure", point.minute), yStart: .value("expected low", point.low),
                            yEnd: .value("expected high", point.high)
                        )
                        .foregroundStyle(Color.accentColor.opacity(0.07))
                    }
                    ForEach(thought) { point in
                        LineMark(
                            x: .value("Departure", point.minute), y: .value("Minutes", point.minutes), series: .value("Series", "expected")
                        )
                        .interpolationMethod(.monotone).foregroundStyle(Color.accentColor.opacity(0.55)).lineStyle(
                            StrokeStyle(lineWidth: 1.5, dash: [5, 3]))
                    }
                    ForEach(fan) { point in
                        AreaMark(
                            x: .value("Departure", point.minute), yStart: .value("10th percentile", point.low),
                            yEnd: .value("90th percentile", point.high)
                        )
                        .foregroundStyle(Color.accentColor.opacity(0.12))
                    }
                    ForEach(fan) { point in
                        LineMark(
                            x: .value("Departure", point.minute), y: .value("Minutes", point.minutes), series: .value("Series", "predicted")
                        ).interpolationMethod(.monotone).foregroundStyle(Color.accentColor).lineStyle(StrokeStyle(lineWidth: 2))
                    }
                    ForEach(aheadPoints) { point in
                        PointMark(x: .value("Departure", point.minute), y: .value("Minutes", point.minutes)).foregroundStyle(.blue)
                            .symbolSize(12)
                    }
                    // A gap in the samples (sampler off, laptop asleep) is left as a gap instead of a straight line over nothing.
                    ForEach(Array(actualSegments(actual).enumerated()), id: \.offset) { index, segment in
                        ForEach(segment) { point in
                            LineMark(
                                x: .value("Departure", point.x), y: .value("Minutes", point.minutes),
                                series: .value("Series", "actual-\(index)")
                            )
                            .interpolationMethod(.monotone).foregroundStyle(Color.red).lineStyle(
                                StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))  // monotone never overshoots the samples
                        }
                        if segment.count == 1, let point = segment.first {
                            PointMark(x: .value("Departure", point.x), y: .value("Minutes", point.minutes)).foregroundStyle(Color.red)
                                .symbolSize(14)
                        }
                    }
                    if let newest = actual.last {
                        PointMark(x: .value("Departure", newest.x), y: .value("Minutes", newest.minutes))
                            .foregroundStyle(Color.red).symbolSize(40)
                    }
                    if let departure = route.shownRecommendation.departAt, let date = parseDate(departure) {
                        let minute = departureMinute(date)
                        RuleMark(x: .value("Suggested departure", minute))
                            .foregroundStyle(Color.green.opacity(0.65))
                            .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                        if let minutes = route.shownRecommendation.expectedMinutes ?? route.shownRecommendation.minutes {
                            PointMark(x: .value("Suggested departure", minute), y: .value("Minutes", minutes))
                                .foregroundStyle(.green).symbolSize(35)
                        }
                    }
                    if let selected {
                        RuleMark(x: .value("Departure", selected.minute)).foregroundStyle(.secondary.opacity(0.5))
                        PointMark(x: .value("Departure", selected.minute), y: .value("Minutes", selected.minutes)).foregroundStyle(.blue)
                            .symbolSize(45)
                    }
                    if let selected = selectedActual {
                        RuleMark(x: .value("Departure", selected.x)).foregroundStyle(.secondary.opacity(0.5))
                        PointMark(x: .value("Departure", selected.x), y: .value("Minutes", selected.minutes)).foregroundStyle(Color.red)
                            .symbolSize(45)
                    }
                }
                .chartXAxis { xAxis(range) }
                .chartXScale(domain: range)
                .chartYScale(
                    domain: yDomain(
                        fan.flatMap { [$0.low, $0.high, $0.minutes] } + actual.map(\.minutes) + thought.flatMap { [$0.low, $0.high] })
                )
                .chartYAxis { AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) }
                .chartOverlay { proxy in hoverCapture(proxy) }
                .frame(height: 145)
                HStack(spacing: 10) {
                    if let newest = actual.last {
                        Text("Red = today\(windowOpen ? " so far" : "") · \(timeLabel(newest.minute)) \(decimal(newest.minutes)) min")
                            .foregroundStyle(Color.red)
                    }
                    if !thought.isEmpty { Text("dashed = expected this morning").foregroundStyle(.secondary) }
                }.font(.caption2).padding(.top, 2)
            }
            if route.modelState == "fallback" {
                Text("The Predicted model scored worse than this simple curve recently, so the simple curve is shown until it does better.")
                    .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).padding(.top, 6)
            } else if route.showsModel, let readout = route.model?.readout, readout.days >= 1 {
                let learning = route.modelState == "calibrating" || readout.days < 15
                Text(
                    "Last \(readout.days) day\(readout.days == 1 ? "" : "s"): avg miss \(oneDecimal(readout.maeMinutes)) min · \(Int((readout.hitRate * 100).rounded()))% of actual drive times inside the range\(learning ? " · still learning" : "")"
                )
                .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).padding(.top, 6)
            }
        }
    }
    /// Both directions across the day on one number line, from the earliest window start to the latest window end;
    /// each direction's window is a fixed tinted box, and samples fill in left to right.
    @ViewBuilder private func todayChart() -> some View {
        let axis = DayAxis(model.snapshot?.routes.values.map { $0 } ?? [])
        let data = todayPoints(axis)
        let range = axis.domain
        let now = localMinute(Date())
        let yRange = yDomain(data.map(\.minutes))
        let ticks = axis.ticks
        // The sample nearest the pointer on each line, so both directions are read out and marked together.
        let selected: [TodayPoint] =
            hoveredMinute.map { x in
                Self.directionLabels.compactMap { _, label in data.filter { $0.label == label }.min { abs($0.x - x) < abs($1.x - x) } }
            } ?? []
        VStack(alignment: .leading, spacing: 4) {
            VStack(alignment: .leading, spacing: 2) {
                if !selected.isEmpty {
                    ForEach(selected) { Text("\(timeLabel($0.minute)) · \(decimal($0.minutes)) min · \($0.label)") }
                } else {
                    Text("Drive time · hover for details")
                }
            }.font(.caption).foregroundStyle(.secondary).frame(height: 32, alignment: .topLeading)
            do {
                Chart {
                    ForEach(Self.directionLabels, id: \.0) { key, label in
                        if let route = model.snapshot?.routes[key] {
                            RectangleMark(
                                xStart: .value("Window start", Double(route.windowStart)),
                                xEnd: .value("Window end", Double(route.windowEnd))
                            )
                            .foregroundStyle((key == "to_work" ? Color.accentColor : Color.orange).opacity(0.12))
                        }
                    }
                    ForEach(data) { point in
                        LineMark(
                            x: .value("Departure", point.x), y: .value("Minutes", point.minutes), series: .value("Direction", point.label)
                        )
                        .interpolationMethod(.monotone).foregroundStyle(by: .value("Direction", point.label)).lineStyle(
                            StrokeStyle(lineWidth: 2))
                        PointMark(x: .value("Departure", point.x), y: .value("Minutes", point.minutes))
                            .foregroundStyle(by: .value("Direction", point.label)).symbolSize(12)
                    }
                    if range.contains(now) {
                        RuleMark(x: .value("Now", now)).foregroundStyle(Color.green).lineStyle(StrokeStyle(lineWidth: 1.5))
                    }
                    if let pointer = hoveredMinute {
                        RuleMark(x: .value("Departure", pointer)).foregroundStyle(.secondary.opacity(0.5))
                        ForEach(selected) { point in
                            PointMark(x: .value("Departure", point.x), y: .value("Minutes", point.minutes))
                                .foregroundStyle(by: .value("Direction", point.label)).symbolSize(55)
                        }
                    }
                }
                .chartForegroundStyleScale(["To work": Color.accentColor, "To home": Color.orange])
                .chartLegend(position: .bottom, alignment: .leading, spacing: 6)
                .chartXAxis {
                    AxisMarks(values: ticks.map(\.minute)) { value in
                        AxisTick()
                        if let minute = value.as(Double.self), ticks.first(where: { $0.minute == minute })?.labeled == true {
                            AxisGridLine().foregroundStyle(.secondary.opacity(0.15))
                            // The end labels hang inward so they are not clipped at the plot edge.
                            AxisValueLabel(
                                anchor: minute == range.lowerBound ? .topLeading : minute == range.upperBound ? .topTrailing : .top
                            ) {
                                Text(timeLabel(minute)).font(.system(size: 9, design: .monospaced)).fixedSize()
                            }
                        }
                    }
                }
                .chartXScale(domain: range)
                .chartYScale(domain: yRange)  // fits the day's drive times; an empty day still draws its axes and windows
                .chartYAxis { AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) }
                .chartOverlay { proxy in hoverCapture(proxy) }
                .frame(height: 165)
                if data.isEmpty {
                    Text("No samples in either window yet today").font(.caption2).foregroundStyle(.secondary)
                } else {
                    // Samples come every ~10 min; a longer hole means the sampler was offline (asleep, no network).
                    let xs = data.map(\.x).sorted()
                    let longestGap = zip(xs, xs.dropFirst()).map { $1 - $0 }.max() ?? 0
                    Text("\(data.count) samples today\(longestGap > 20 ? " · longest gap \(durationText(Int(longestGap)))" : "")")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }
    /// A scored day on the report-card charts.
    private struct ScorePoint: Identifiable {
        let id: String  // the local day, yyyy-MM-dd
        let day: Date
        let miss: Double  // average miss of the model's expected drive time that day (minutes)
        let inside: Double  // share of real drives that landed inside the model's 80% range (0...1)
        let n: Int  // forecast targets scored that day
        let score: Double?  // interval score of the 80% band — the number the model itself is judged on
    }
    private struct CmpPoint: Identifiable {
        let id: String
        let day: Date
        let miss: Double  // the simple curve's average miss on the same day
        let score: Double?  // the simple curve's interval score that day
    }
    /// Every model-tab chart shares this label width so the y-axis gutters — and therefore every
    /// plot frame, gridline, and date tick — line up vertically.
    private static let modelAxisLabelWidth: CGFloat = 30
    private func modelAxisLabel(_ text: String) -> some View {
        Text(text).font(.caption2).frame(width: Self.modelAxisLabelWidth, alignment: .trailing)
    }
    /// A day-scale domain padded past the outermost weekly tick labels so edge dates are never clipped.
    private func paddedDayDomain(_ days: [Date]) -> ClosedRange<Date> {
        guard let first = days.min(), let last = days.max(), last > first else {
            let today = days.first ?? Date()
            return today.addingTimeInterval(-2 * 86400)...today.addingTimeInterval(2 * 86400)
        }
        let pad = max(2.5 * 86400, last.timeIntervalSince(first) * 0.06)
        return first.addingTimeInterval(-pad)...last.addingTimeInterval(pad)
    }
    /// The report-card series plus the model's edge over the simple curve, derived outside the view builder.
    /// The edge compares interval scores — what the fallback gate itself uses — and falls back to misses.
    private func scoreSummary(_ info: RouteModel?) -> (days: [ScorePoint], simple: [CmpPoint], edge: Double?) {
        var days: [ScorePoint] = [], simple: [CmpPoint] = []
        for day in info?.series ?? [] {
            guard let date = dayDate(day.date) else { continue }
            days.append(ScorePoint(id: day.id, day: date, miss: day.mae, inside: day.cov80, n: day.n ?? 0, score: day.is80))
            if let miss = day.cmpMae {
                simple.append(CmpPoint(id: day.id, day: date, miss: miss, score: day.cmpIs80))
            }
        }
        let simpleByDay = Dictionary(simple.map { ($0.id, $0) }, uniquingKeysWith: { _, new in new })
        var modelScore = 0.0, simpleScore = 0.0, modelMiss = 0.0, simpleMiss = 0.0, scoreDays = 0, missDays = 0
        for day in days {
            guard let cmp = simpleByDay[day.id] else { continue }
            if let model = day.score, let theirs = cmp.score {
                modelScore += model
                simpleScore += theirs
                scoreDays += 1
            }
            modelMiss += day.miss
            simpleMiss += cmp.miss
            missDays += 1
        }
        let edge: Double? =
            scoreDays >= 3 && simpleScore > 0
            ? 100 * (1 - modelScore / simpleScore)
            : missDays >= 3 && simpleMiss > 0 ? 100 * (1 - modelMiss / simpleMiss) : nil
        return (days, simple, edge)
    }
    /// The model's report card. Every finished day is scored on how far the expected drive times missed the real
    /// ones and how often reality landed inside the stated range; the simple curve gets the same scores, so these
    /// charts are also the race the model must keep winning to stay on screen.
    @ViewBuilder private func modelTab(_ route: Route) -> some View {
        let info = route.model
        let (days, simple, edge) = scoreSummary(info)
        let selected = hoveredDay.flatMap { date in days.min { abs($0.day.timeIntervalSince(date)) < abs($1.day.timeIntervalSince(date)) } }
        let selectedCmp = selected.flatMap { s in simple.first { $0.id == s.id } }
        // One shared day domain for all three charts — identical spans keep the grids aligned.
        let dayDomain = paddedDayDomain(days.map(\.day))
        VStack(alignment: .leading, spacing: 4) {
            VStack(alignment: .leading, spacing: 2) {
                if let selected {
                    Text(
                        "\(selected.day.formatted(.dateTime.weekday(.abbreviated).month(.abbreviated).day())) · missed by \(oneDecimal(selected.miss)) min\(selectedCmp.map { " · simple curve \(oneDecimal($0.miss))" } ?? "")"
                    )
                    Text("\(Int((selected.inside * 100).rounded()))% of \(selected.n) checks inside the range")
                } else if let readout = info?.readout, readout.days >= 1 {
                    Text(
                        "Avg miss \(oneDecimal(readout.maeMinutes)) min · \(Int((readout.hitRate * 100).rounded()))% of drives inside the range"
                    )
                    Text("\(info?.ledgerDays ?? 0) days scored\(edge.map { " · \(edgeText($0))" } ?? "")")
                } else {
                    Text("No scored days yet")
                }
            }.font(.caption).foregroundStyle(.secondary).frame(height: 32, alignment: .topLeading)
            if days.isEmpty {
                Text("The model starts building a track record once each finished day is scored.")
                    .foregroundStyle(.secondary).multilineTextAlignment(.center).frame(height: 145).frame(maxWidth: .infinity)
            } else {
                Text("Average miss per scored day").font(.caption).padding(.top, 6)
                Chart {
                    ForEach(simple) { point in
                        LineMark(x: .value("Day", point.day, unit: .day), y: .value("Miss", point.miss), series: .value("s", "simple"))
                            .foregroundStyle(.secondary).lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                    }
                    ForEach(days) { point in
                        LineMark(x: .value("Day", point.day, unit: .day), y: .value("Miss", point.miss), series: .value("s", "model"))
                            .foregroundStyle(Color.accentColor).lineStyle(StrokeStyle(lineWidth: 2))
                        PointMark(x: .value("Day", point.day, unit: .day), y: .value("Miss", point.miss))
                            .foregroundStyle(Color.accentColor).symbolSize(12)
                    }
                    if let selected {
                        RuleMark(x: .value("Day", selected.day, unit: .day)).foregroundStyle(.secondary.opacity(0.5))
                    }
                }
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 4)) }
                .chartXScale(domain: dayDomain)
                .chartYScale(domain: yDomain(days.map(\.miss) + simple.map(\.miss)))
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                        AxisGridLine()
                        AxisValueLabel { modelAxisLabel("\(Int((value.as(Double.self) ?? 0).rounded()))") }
                    }
                }
                .chartOverlay { proxy in modelHoverCapture(proxy) }
                .frame(height: 100)
                if let byWeekday = weekdayMissCaption(days) {
                    Text(byWeekday).font(.caption2).foregroundStyle(.secondary)
                }
                Text("Real drives landing inside the \(info?.intervalPercent ?? 80)% range").font(.caption).padding(.top, 10)
                Chart {
                    RuleMark(y: .value("Target", (info?.gates?.cov ?? 0.8) * 100))
                        .foregroundStyle(.green.opacity(0.65)).lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    ForEach(days) { point in
                        LineMark(
                            x: .value("Day", point.day, unit: .day), y: .value("Inside", point.inside * 100), series: .value("s", "inside")
                        )
                        .foregroundStyle(Color.accentColor).lineStyle(StrokeStyle(lineWidth: 2))
                        PointMark(x: .value("Day", point.day, unit: .day), y: .value("Inside", point.inside * 100))
                            .foregroundStyle(Color.accentColor).symbolSize(12)
                    }
                    if let selected {
                        RuleMark(x: .value("Day", selected.day, unit: .day)).foregroundStyle(.secondary.opacity(0.5))
                    }
                }
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 4)) }
                .chartXScale(domain: dayDomain)
                .chartYScale(domain: 0...100)
                .chartYAxis {
                    AxisMarks(position: .leading, values: [0, 50, 100]) { value in
                        AxisGridLine()
                        AxisValueLabel { modelAxisLabel("\(Int((value.as(Double.self) ?? 0).rounded()))%") }
                    }
                }
                .chartOverlay { proxy in modelHoverCapture(proxy) }
                .frame(height: 80)
            }
            let (params, paramKeys) = modelParams(info)
            if params.count >= 4 {
                Text("What it leans on").font(.caption).padding(.top, 10)
                // Weight chips double as the chart legend: one signed chip per line; the
                // square-marked rain chip carries its own min/mm unit and has no line by design.
                let chips = signalChips(info, charted: Set(paramKeys.map { $0.key }))
                let weightRange = weightDomain(params)
                if !chips.isEmpty {
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 3) {
                        GridRow {
                            signalChip(chips[0])
                            signalChip(chips.count > 1 ? chips[1] : nil)
                        }
                        if chips.count > 2 {
                            GridRow {
                                signalChip(chips[2])
                                signalChip(chips.count > 3 ? chips[3] : nil)
                            }
                        }
                    }
                }
                Chart {
                    if weightRange.lowerBound < 0 {
                        RuleMark(y: .value("Zero", 0))
                            .foregroundStyle(.secondary.opacity(0.6)).lineStyle(StrokeStyle(lineWidth: 0.75))
                    }
                    ForEach(paramKeys.indices, id: \.self) { i in
                        // Coefficients hold between refits, so steps are honest: the value at each day
                        // is the weight the model actually used until the next scored day replaced it.
                        ForEach(params.filter { $0.key == paramKeys[i].key }) { point in
                            LineMark(
                                x: .value("Day", point.day, unit: .day), y: .value("Weight", point.value * 100),
                                series: .value("Signal", paramKeys[i].label)
                            )
                            .foregroundStyle(paramKeys[i].color).lineStyle(StrokeStyle(lineWidth: 1.5))
                            .interpolationMethod(.stepEnd)
                        }
                    }
                    if let selected {
                        RuleMark(x: .value("Day", selected.day, unit: .day)).foregroundStyle(.secondary.opacity(0.5))
                    }
                }
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 4)) }
                .chartXScale(domain: dayDomain)
                // Include negative weights and pad the chart beyond the observed extremes.
                .chartYScale(domain: weightRange)
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { value in
                        AxisGridLine()
                        AxisValueLabel { modelAxisLabel(percentText(value.as(Double.self) ?? 0)) }
                    }
                }
                .chartOverlay { proxy in modelHoverCapture(proxy) }
                .frame(height: 85)
            }
            if let swing = swingCaption(info) {
                Text(swing).font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true).padding(.top, 6)
            }
            if let progress = modelProgress(info) {
                Text(progress).font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }
    /// Where the model sits on the way to an earned "Predicted" label; nil once it has arrived.
    private func modelProgress(_ info: RouteModel?) -> String? {
        let scored = info?.ledgerDays ?? 0
        switch info?.state {
        case "collecting":
            return "Still collecting data — the forecast starts after about 3 commuting days."
        case "learning":
            return "Building its track record — the learned curve takes over around day \(info?.gates?.cal ?? 5) (day \(scored) now)."
        case "calibrating":
            return "Proving itself — day \(scored) of ~\(info?.gates?.ready ?? 15) needed for the Predicted label."
        case "fallback":
            return "Off screen — the simple curve scored better recently; it comes back when it wins again."
        default:
            return nil
        }
    }
    /// One point of the learned-parameter history: a weight the model was using on that scored day.
    private struct ParamPoint: Identifiable {
        let id: String
        let day: Date
        let key: String
        let value: Double
    }
    /// The weight chart's % range: up to a bit above the largest weight, and down to 0 — or a little below the most
    /// negative weight, so a line that dips under zero stays inside the plot instead of running into the date labels.
    private func weightDomain(_ points: [ParamPoint]) -> ClosedRange<Double> {
        let percents = points.map { $0.value * 100 }
        let low = min(0, percents.min() ?? 0), high = max(50, (percents.max() ?? 0) * 1.15)
        return (low < 0 ? low * 1.15 - 3 : 0)...high
    }
    /// One color-coded signal in the "what it leans on" legend: label, current value, its color, its
    /// plain-language explanation, and whether it is a weight (a % with a chart line) or a rate
    /// (its own unit, a square marker).
    private struct SignalChip: Identifiable {
        let id: String
        let label: String
        let value: String
        let color: Color
        let help: String
        let isWeight: Bool
    }
    /// One signed weight chip for each charted line, with its current or latest recorded value;
    /// rain as a square-marked chip in its own min/mm unit so no unit is ever silently mixed.
    private func signalChips(_ info: RouteModel?, charted: Set<String>) -> [SignalChip] {
        let cov = info?.cov ?? [:]
        var chips: [SignalChip] = []
        for (key, label, color) in paramDefs() {
            switch key {
            case "phi":
                if charted.contains(key),
                    let v = info?.phi ?? info?.series?.last(where: { $0.learned?[key] != nil })?.learned?[key]
                {
                    chips.append(
                        SignalChip(
                            id: key, label: label, value: percentText(v * 100), color: color,
                            help: "Share of the day-to-day swing explained by which weekday it is.", isWeight: true))
                }
            case "sib":
                if charted.contains(key),
                    let v = cov[key] ?? info?.series?.last(where: { $0.learned?[key] != nil })?.learned?[key]
                {
                    chips.append(
                        SignalChip(
                            id: key, label: label, value: percentText(v * 100), color: color,
                            help:
                                "How much of the other direction's latest traffic carries into this drive. Below zero it works the other way: a slow one points to a slightly faster drive.",
                            isWeight: true))
                }
            case "prev":
                if charted.contains(key),
                    let v = cov[key] ?? info?.series?.last(where: { $0.learned?[key] != nil })?.learned?[key]
                {
                    chips.append(
                        SignalChip(
                            id: key, label: label, value: percentText(v * 100), color: color,
                            help:
                                "How much of the previous day's leftover traffic carries into today. Below zero it works the other way: a slow day points to a slightly faster one.",
                            isWeight: true))
                }
            default:  // rain is a minutes-per-mm rate, not a weight — own unit, own marker shape
                if let r = cov[key], abs(r) >= 0.005 {
                    chips.append(
                        SignalChip(
                            id: key, label: label, value: "\(r >= 0 ? "+" : "−")\(String(format: "%.2f", abs(r))) min/mm", color: color,
                            help:
                                "Minutes the model adds to the day's forecast per mm of forecast rain — about \(r >= 0 ? "+" : "−")\(String(format: "%.1f", abs(r * 5))) min on a 5 mm day.",
                            isWeight: false))
                }
            }
        }
        return chips
    }
    /// A legend chip: colored dot, colored signal name, gray weight.
    private func signalChip(_ chip: SignalChip?) -> some View {
        Group {
            if let chip {
                HStack(spacing: 5) {
                    if chip.isWeight {
                        Circle().fill(chip.color).frame(width: 6, height: 6)
                    } else {
                        RoundedRectangle(cornerRadius: 1.5).fill(chip.color).frame(width: 6, height: 6)
                    }
                    Text(chip.label).font(.caption2).foregroundStyle(chip.color)
                    Text(chip.value).font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                }
                .help(chip.help)
            } else {
                Text(" ").font(.caption2)
            }
        }
    }
    /// Average miss broken out per weekday — the weekdays where the model is weakest, at a glance.
    /// A weekday needs at least two scored days before a lone bad day counts as a pattern.
    private func weekdayMissCaption(_ days: [ScorePoint]) -> String? {
        var sums: [Int: (total: Double, n: Int)] = [:]
        for day in days {
            let w = Calendar.current.component(.weekday, from: day.day)
            let cur = sums[w] ?? (0, 0)
            sums[w] = (cur.total + day.miss, cur.n + 1)
        }
        let symbols = Calendar.current.shortWeekdaySymbols
        let parts = (1...7).compactMap { w -> String? in
            guard let (total, n) = sums[w], n >= 2 else { return nil }
            return "\(symbols[w - 1]) \(oneDecimal(total / Double(n)))"
        }
        return parts.count >= 3 ? "Avg miss by weekday: " + parts.joined(separator: " · ") : nil
    }
    /// How much day-to-day variability the model sees, and how much of it nothing it watches explains.
    private func swingCaption(_ info: RouteModel?) -> String? {
        guard let tau = info?.tau, tau > 0 else { return nil }
        var swing = "Day-to-day swing ±\(oneDecimal(tau)) min"
        if let ve2 = info?.cov?["ve2"], ve2 > 0, ve2 < tau * tau {
            swing += " · ±\(oneDecimal(ve2.squareRoot())) unexplained"
        }
        return swing
    }
    /// The (key, label, color) table the signal chips and the weight chart share, so they never disagree.
    private func paramDefs() -> [(key: String, label: String, color: Color)] {
        let sibLabel = model.direction == "to_home" ? "morning" : "last night"
        return [
            ("phi", "weekday", .accentColor), ("sib", sibLabel, .green),
            ("prev", "yesterday", .orange), ("rain", "rain", .purple)
        ]
    }
    /// The learned-parameter history from the scoring ledger: how the weights evolved day by day.
    /// Only fractional weights are charted — rain is a minutes-per-mm rate, not a weight, so it stays
    /// on the chips and out of a % axis that would mislabel it.
    private func modelParams(_ info: RouteModel?) -> (points: [ParamPoint], keys: [(key: String, label: String, color: Color)]) {
        let defs = paramDefs().filter { $0.key != "rain" }
        var points: [ParamPoint] = []
        for day in info?.series ?? [] {
            guard let date = dayDate(day.date), let learned = day.learned else { continue }
            for (key, _, _) in defs {
                if let value = learned[key] {
                    points.append(ParamPoint(id: "\(day.id)-\(key)", day: date, key: key, value: value))
                }
            }
        }
        let keys = defs.filter { key, _, _ in points.contains { $0.key == key } }
        return (points, keys)
    }
    private func edgeText(_ edge: Double) -> String {
        edge >= 0 ? "~\(Int(edge.rounded()))% better than the simple curve" : "~\(Int((-edge).rounded()))% worse than the simple curve"
    }
}
