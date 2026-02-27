/** Shared placeholder used by all unbuilt pages. */
export default function Placeholder({ title }: { title: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center gap-4">
      <p className="text-5xl">🚧</p>
      <h2 className="text-2xl font-bold text-[#002B5B]">{title}</h2>
      <p className="text-gray-500">Coming soon — full migration in progress.</p>
    </div>
  )
}
