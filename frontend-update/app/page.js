// "use client";

// import Image from "next/image";
// import Link from "next/link";
// import { motion } from "framer-motion";
// import * as React from "react";
// import {
//   NavigationMenu,
//   NavigationMenuItem,
//   NavigationMenuLink,
//   NavigationMenuList,
// } from "@/components/ui/navigation-menu";

// export default function Home() {
//   return (
//     <div className="min-h-screen flex flex-col">
//       {/* Navbar */}
//       <div className="border-b sticky top-0 z-50 bg-white">
//         <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4">
//           {/* Left side: Brand + Logo */}
//           <div className="flex items-center gap-2">
//             <Image
//               src="/underwriting.png"
//               alt="Acquire Underwriting Logo"
//               width={50}
//               height={50}
//             />
//             <span className="text-lg font-bold">Acquire Underwriting</span>
//           </div>

//           {/* Right side: Nav links */}
//           <NavigationMenu>
//             <NavigationMenuList className="flex items-center gap-6">
//               <NavigationMenuItem>
//                 <NavigationMenuLink asChild>
//                   <Link href="/" className="font-medium hover:text-primary">
//                     Home
//                   </Link>
//                 </NavigationMenuLink>
//               </NavigationMenuItem>
//               <NavigationMenuItem>
//                 <NavigationMenuLink asChild>
//                   <Link
//                     href="/signin"
//                     className="rounded-lg border bg-primary px-4 py-2 text-white hover:bg-primary/90 transition"
//                   >
//                     Sign In
//                   </Link>
//                 </NavigationMenuLink>
//               </NavigationMenuItem>
//             </NavigationMenuList>
//           </NavigationMenu>
//         </div>
//       </div>

//       {/* Hero Section */}
//       <motion.section
//         initial={{ opacity: 0, y: 40 }}
//         animate={{ opacity: 1, y: 0 }}
//         transition={{ duration: 0.8 }}
//         className="flex flex-1 items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 px-6 py-20 text-center"
//       >
//         <div className="max-w-3xl">
//           <motion.h1
//             initial={{ opacity: 0, y: 30 }}
//             animate={{ opacity: 1, y: 0 }}
//             transition={{ delay: 0.2, duration: 0.7 }}
//             className="text-4xl font-extrabold sm:text-6xl"
//           >
//             Smarter <span className="text-primary">Underwriting</span> Decisions
//           </motion.h1>
//           <motion.p
//             initial={{ opacity: 0 }}
//             animate={{ opacity: 1 }}
//             transition={{ delay: 0.5, duration: 0.8 }}
//             className="mt-6 text-lg text-gray-600"
//           >
//             Acquire Underwriting helps you analyze, summarize, and make confident
//             investment decisions with automated insights powered by AI.
//           </motion.p>
//           <motion.div
//             initial={{ opacity: 0, scale: 0.9 }}
//             animate={{ opacity: 1, scale: 1 }}
//             transition={{ delay: 0.8, duration: 0.6 }}
//             className="mt-8 flex justify-center gap-4"
//           >
//             <Link
//               href="/signup"
//               className="rounded-lg bg-primary px-6 py-3 text-white font-medium hover:bg-primary/90 transition"
//             >
//               Get Started
//             </Link>
//             <Link
//               href="/learn-more"
//               className="rounded-lg border px-6 py-3 font-medium hover:bg-gray-100 transition"
//             >
//               Learn More
//             </Link>
//           </motion.div>
//         </div>
//       </motion.section>

//       {/* Features Section */}
//       <motion.section
//         initial={{ opacity: 0, y: 40 }}
//         whileInView={{ opacity: 1, y: 0 }}
//         transition={{ duration: 0.8 }}
//         viewport={{ once: true }}
//         className="mx-auto max-w-6xl px-6 py-20"
//       >
//         <h2 className="text-3xl font-bold text-center">
//           Why Choose Acquire Underwriting?
//         </h2>
//         <div className="mt-12 grid gap-8 md:grid-cols-3">
//           {[
//             {
//               title: "Automated Analysis",
//               desc: "Quickly generate underwriting summaries with AI-powered insights.",
//               icon: "📊",
//             },
//             {
//               title: "Financial Metrics",
//               desc: "Instantly calculate NOI, Cap Rate, IRR and Rent Gaps from raw data.",
//               icon: "💰",
//             },
//             {
//               title: "Smart Decisions",
//               desc: "Get clear buy/hold/sell recommendations with professional-grade reports.",
//               icon: "🎯",
//             },
//           ].map((feature, i) => (
//             <motion.div
//               key={feature.title}
//               initial={{ opacity: 0, y: 40 }}
//               whileInView={{ opacity: 1, y: 0 }}
//               transition={{ delay: i * 0.2, duration: 0.7 }}
//               viewport={{ once: true }}
//               className="rounded-2xl border bg-white p-6 shadow-sm hover:shadow-md transition"
//             >
//               <div className="text-4xl">{feature.icon}</div>
//               <h3 className="mt-4 text-xl font-semibold">{feature.title}</h3>
//               <p className="mt-2 text-gray-600">{feature.desc}</p>
//             </motion.div>
//           ))}
//         </div>
//       </motion.section>
//     </div>
//   );
// }



"use client";

import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import * as React from "react";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from "@/components/ui/navigation-menu";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Navbar */}
      <div className="border-b sticky top-0 z-50 bg-white/80 backdrop-blur-md shadow-sm">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4">
          {/* Left side: Brand + Logo */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="flex items-center gap-2"
          >
            <Image
              src="/underwriting.png"
              alt="Acquire Underwriting Logo"
              width={50}
              height={50}
              className="drop-shadow-md"
            />
            <span className="text-lg font-bold">Acquire Underwriting</span>
          </motion.div>

          {/* Right side: Nav links */}
          <NavigationMenu>
            <NavigationMenuList className="flex items-center gap-6">
              <NavigationMenuItem>
                <NavigationMenuLink asChild>
                  <Link href="/" className="font-medium hover:text-primary">
                    Home
                  </Link>
                </NavigationMenuLink>
              </NavigationMenuItem>
              <NavigationMenuItem>
                <NavigationMenuLink asChild>
                  <Link
                    href="/auth"
                    className="rounded-lg border bg-primary px-4 py-2 text-white hover:bg-primary/90 transition"
                  >
                    Sign In
                  </Link>
                </NavigationMenuLink>
              </NavigationMenuItem>
            </NavigationMenuList>
          </NavigationMenu>
        </div>
      </div>

      {/* Hero Section */}
      <motion.section
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeOut" }}
        className="flex flex-1 items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 px-6 py-20 text-center"
      >
        <div className="max-w-3xl">
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="text-5xl font-extrabold sm:text-6xl drop-shadow-md"
          >
            Smarter <span className="text-primary">Underwriting</span> Decisions
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.9 }}
            className="mt-6 text-lg text-gray-600"
          >
            Acquire Underwriting helps you analyze, summarize, and make confident
            investment decisions with automated insights powered by AI.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.8, duration: 0.7 }}
            className="mt-8 flex justify-center gap-4"
          >
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link
                href="/auth"
                className="rounded-lg bg-primary px-6 py-3 text-white font-medium shadow-md hover:shadow-lg transition"
              >
                Get Started
              </Link>
            </motion.div>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link
                href="/learn-more"
                className="rounded-lg border px-6 py-3 font-medium shadow-md hover:bg-gray-100 transition"
              >
                Learn More
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </motion.section>

      {/* Features Section */}
      <motion.section
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
        className="mx-auto max-w-6xl px-6 py-20"
      >
        <h2 className="text-3xl font-bold text-center drop-shadow-sm">
          Why Choose Acquire Underwriting?
        </h2>
        <div className="mt-12 grid gap-8 md:grid-cols-3">
          {[
            {
              title: "Automated Analysis",
              desc: "Quickly generate underwriting summaries with AI-powered insights.",
              icon: "📊",
            },
            {
              title: "Financial Metrics",
              desc: "Instantly calculate NOI, Cap Rate, IRR and Rent Gaps from raw data.",
              icon: "💰",
            },
            {
              title: "Smart Decisions",
              desc: "Get clear buy/hold/sell recommendations with professional-grade reports.",
              icon: "🎯",
            },
          ].map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 40, rotateX: -15 }}
              whileInView={{ opacity: 1, y: 0, rotateX: 0 }}
              whileHover={{
                scale: 1.05,
                rotateX: 5,
                rotateY: 5,
                boxShadow: "0px 10px 30px rgba(0,0,0,0.1)",
              }}
              whileTap={{ scale: 0.97 }}
              transition={{ delay: i * 0.2, duration: 0.7, ease: "easeOut" }}
              viewport={{ once: true }}
              className="rounded-2xl border bg-white p-6 shadow-md hover:shadow-xl transition transform-gpu"
            >
              <div className="text-4xl">{feature.icon}</div>
              <h3 className="mt-4 text-xl font-semibold">{feature.title}</h3>
              <p className="mt-2 text-gray-600">{feature.desc}</p>
            </motion.div>
          ))}
        </div>
      </motion.section>
    </div>
  );
}
